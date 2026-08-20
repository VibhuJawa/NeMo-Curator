# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The assets an extracted document points at: keeping them, and finding their bytes.

Two independent problems, both about `<img src>`:

**Keeping them.** The converter emits at most one image per ``<figure>``. Measured
against the real converter: 1 of 2 and 1 of 3 kept, whichever way the images are nested
inside the figure — same ``<span>``, same ``<p>``, separate ``<p>``s — while two images
in two ``<figure>`` elements both survive. :func:`split_multi_image_figures` gives each
image its own figure before conversion, which recovers all of them.

**Finding their bytes.** A document's markup names an asset the way the document was
written — ``x1.png``, relative to wherever the document came from. That is not something
a downstream reader can fetch, and absolutising it against the page URL manufactures a
URL that never existed (``https://arxiv.org/abs/x1.png``, from a page URL with no
trailing segment to replace). An :class:`AssetResolver` maps ``(document, src)`` onto the
archive member that actually holds the bytes, and the interleaved stage records that as
a ``source_ref`` locator the existing materialization path already understands.

Nothing here knows what corpus it is looking at. Which column is the document id, which
archive a document's assets live in, and how a member is named are all configuration the
driver passes.
"""

from __future__ import annotations

import abc
import re
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.stages.text.html_extraction.mineru_utils import element_to_html, html_to_element
from nemo_curator.tasks import InterleavedBatch

if TYPE_CHECKING:
    from lxml.html import HtmlElement

# Element -> the attribute naming an external asset. `<img src>` is all these documents
# have, and all that reaches the interleaved rows. `<source srcset>` and `<object data>`
# are one entry each plus a parser for that attribute's own syntax — srcset is a
# comma-separated candidate list, not a single reference — which is why this is a table
# and not an `if tag == "img"` buried in the walk below.
ASSET_ATTRIBUTES: dict[str, str] = {"img": "src"}

FIGURE_TAG = "figure"

# A src the converter left alone: it already carries a scheme, or is protocol-relative.
# Those name bytes somewhere else entirely, so there is no archive member to look up.
_ABSOLUTE_SRC_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.\-]*:|//)")


def is_absolute_src(src: str) -> bool:
    """True for ``https://…``, ``s3://…``, ``data:…``, ``//host/…`` — anything not relative."""
    return _ABSOLUTE_SRC_RE.match(src) is not None


# -- (a) one image per figure ------------------------------------------------------


def split_multi_image_figures(html: str) -> str:
    """Give every image in a ``<figure>`` its own figure, in document order.

    Workaround for a bug in the third-party converter (``mineru-webkit``'s
    ``webpage_converter``), which emits at most one image per figure: its image
    recogniser keeps a single ``<img>`` per figure element and discards the rest.
    Measured directly against it — 1 of 2 and 1 of 3 images kept, unchanged by how they
    nest inside the figure, and 2 of 2 once they sit in two separate figures; on a real
    document (arXiv 1509.05029) 9 ``<img>`` in 6 ``<figure>`` became at most 6 images in
    the markdown. Re-shaped this way the same fixtures convert 3 of 3.

    **Delete this function if the converter learns to keep them.** It exists only to
    compensate, and it is a no-op on documents whose figures hold one image each.

    The caption is not duplicated: the leading images move out into bare figures and the
    last one stays behind with the original figure's ``<figcaption>`` and everything else
    it held. Rows therefore arrive as image, image, caption in document order, which is
    the association the reader is left to make.

    Args:
        html: Main-content HTML.

    Returns:
        The HTML, unchanged (identical object) when no figure holds more than one
        image — which is the common case and must not pay for a re-serialisation.
    """
    # Cheap rejection first: this runs on every document, and re-parsing one to
    # discover it has no figures at all would be the whole cost of the workaround.
    # The pipeline serialises its HTML with lxml, so the tag is lower-case here.
    if not html or f"<{FIGURE_TAG}" not in html:
        return html

    root = html_to_element(html)

    # Group by *nearest* figure ancestor, not by "any image under this figure":
    # sub-figures are nested figures, and each already owns exactly one image. Pulling
    # those out would be splitting a figure that the converter handles correctly.
    owned: dict[HtmlElement, list[HtmlElement]] = {}
    for element in root.iter(*ASSET_ATTRIBUTES):
        figure = next(element.iterancestors(FIGURE_TAG), None)
        if figure is not None:
            owned.setdefault(figure, []).append(element)

    changed = False
    for figure, elements in owned.items():
        if len(elements) < 2:  # noqa: PLR2004 - "more than one image" is the whole condition
            continue
        parent = figure.getparent()
        if parent is None or figure is root:
            # The figure is the whole fragment, so it has nothing to become a sibling
            # of: lxml keeps its implicit wrapper outside the root it hands back, and
            # anything inserted beside the figure would be dropped by the serialiser
            # rather than kept. Leave it whole. Only reachable from a hand-made
            # fragment — extracted main content is always a document.
            logger.debug("Figure with multiple images is the whole fragment; left as it is")
            continue
        at = parent.index(figure)
        for offset, element in enumerate(elements[:-1]):
            shell = _empty_copy(figure)
            shell.append(_lift_out(element, figure))
            parent.insert(at + offset, shell)
        changed = True

    return element_to_html(root) if changed else html


def _empty_copy(element: HtmlElement) -> HtmlElement:
    """The element's tag and attributes, no children — minus ``id``, which is unique."""
    return element.makeelement(element.tag, {k: v for k, v in element.attrib.items() if k != "id"})


def _detach(element: HtmlElement) -> None:
    """Remove *element* from its parent, leaving its tail text where it was.

    lxml hangs the text *after* an element off that element, so removing a node the
    naive way silently deletes the prose that followed it.
    """
    parent = element.getparent()
    if element.tail:
        previous = element.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + element.tail
        else:
            parent.text = (parent.text or "") + element.tail
        element.tail = None
    parent.remove(element)


def _lift_out(element: HtmlElement, figure: HtmlElement) -> HtmlElement:
    """Detach *element*, wrapped in empty copies of its ancestors below *figure*.

    An image inside ``<figure><p class="ltx_p"><span class="ltx_text">`` keeps that
    wrapping in its new figure. The converter's output depends on what an image sits in
    — a bare ``<img>`` directly under a figure is a shape these documents never contain
    — so the workaround reproduces the context rather than inventing a simpler one.

    The chain is read before the detach, not after: an element with no parent has no
    ancestors to read, so doing it the other way round silently unwraps every image.
    """
    ancestors: list[HtmlElement] = []
    for ancestor in element.iterancestors():
        if ancestor is figure:
            break
        ancestors.append(ancestor)

    _detach(element)

    node = element
    for ancestor in ancestors:
        shell = _empty_copy(ancestor)
        shell.append(node)
        node = shell
    return node


# -- (b) where the bytes are -------------------------------------------------------


@dataclass(frozen=True)
class AssetLocator:
    """Where an asset's bytes are: a path, and optionally a member and a byte range."""

    path: str
    member: str | None = None
    byte_offset: int | None = None
    byte_size: int | None = None

    def to_source_ref(self) -> str:
        """The locator as the schema's ``source_ref`` JSON string."""
        return InterleavedBatch.build_source_ref(
            path=self.path,
            member=self.member,
            byte_offset=self.byte_offset,
            byte_size=self.byte_size,
        )


class AssetResolver(abc.ABC):
    """Maps a document's reference to an asset onto the bytes that reference names.

    Implementations differ only in where the mapping comes from — read from the archive
    on first touch, or precomputed — never in what comes out. Both must return a locator
    the interleaved schema can carry and materialization can fetch, or ``None`` when the
    asset is not there, which degrades one row rather than failing the document.
    """

    @abc.abstractmethod
    def resolve(self, doc_id: str, src: str, archive: str | None = None) -> AssetLocator | None:
        """Locate *src* as referenced by document *doc_id*.

        Args:
            doc_id: The document the reference was found in.
            src: The reference, exactly as the markup wrote it (``x1.png``).
            archive: Opaque name of the archive the document's assets live in, taken
                from a column the driver names. Implementations that keep their own
                index of paths ignore it.

        Returns:
            A locator, or ``None`` if this resolver has no bytes for that reference.
        """


def _normalise_src(src: str) -> str:
    """The reference as an archive would have stored it.

    Only what is certainly not part of a filename: a leading ``./``, and a query or
    fragment. Percent-escapes are left alone — un-escaping would corrupt any member
    whose name genuinely contains a ``%``.
    """
    src = src.split("?", 1)[0].split("#", 1)[0]
    while src.startswith("./"):
        src = src[2:]
    return src


class TarAssetResolver(AssetResolver):
    """Read the tar's member headers on first touch, then answer from memory.

    No index to build and nothing to keep in step with the data: the offsets come from
    the archive itself. The scan walks header to header, skipping the payloads, so it
    costs one pass of seeks rather than a read of the archive — but it is still a pass,
    which is why the result is cached per archive and why
    :class:`ParquetIndexAssetResolver` exists for runs that would rather pay it once.
    """

    def __init__(
        self,
        archive: str | None = None,
        archive_template: str | None = None,
        member_template: str = "{doc_id}/{src}",
        max_indexed_archives: int = 4,
    ):
        """
        Args:
            archive: Path of the one tar holding every asset, when there is only one.
            archive_template: Path of the tar for a document's ``archive`` value, as a
                format string with an ``{archive}`` field — e.g.
                ``"/data/assets/{archive}/part-0000.tar"``. Takes precedence over
                *archive*. This is where "which tar a shard's assets are in" lives, and
                it is the driver's business, not the stage's.
            member_template: How a member is named, from ``{doc_id}`` and ``{src}``.
                The default namespaces each document's assets under its own id.
            max_indexed_archives: How many archives' indices to keep. A partition
                touches one archive; the bound exists so a worker that wanders across
                many does not accumulate all of them.
        """
        self.archive = archive
        self.archive_template = archive_template
        self.member_template = member_template
        self.max_indexed_archives = max_indexed_archives
        self._indices: OrderedDict[str, dict[str, tuple[int, int]]] = OrderedDict()

    def __getstate__(self) -> dict[str, object]:
        # The index is worker-local and can be tens of thousands of entries. Shipping it
        # to every worker as part of the stage would be paying for it once per worker
        # and once more per pickle.
        state = self.__dict__.copy()
        state["_indices"] = OrderedDict()
        return state

    def archive_path(self, archive: str | None) -> str | None:
        """The tar to look in for a document whose archive column says *archive*."""
        if self.archive_template is not None:
            return self.archive_template.format(archive=archive) if archive else None
        return self.archive

    def index(self, path: str) -> dict[str, tuple[int, int]]:
        """``{member: (byte_offset, byte_size)}`` for one archive, read once."""
        if path in self._indices:
            self._indices.move_to_end(path)
            return self._indices[path]

        try:
            with tarfile.open(path) as archive:
                index = {member.name: (member.offset_data, member.size) for member in archive if member.isfile()}
        except (OSError, tarfile.TarError) as exc:
            # Cached as empty on purpose: an unreadable archive degrades its rows, and
            # says so once, instead of re-opening the same broken file per image.
            logger.warning(f"Cannot index asset archive {path}: {exc}")
            index = {}

        self._indices[path] = index
        while len(self._indices) > self.max_indexed_archives:
            self._indices.popitem(last=False)
        return index

    def resolve(self, doc_id: str, src: str, archive: str | None = None) -> AssetLocator | None:
        path = self.archive_path(archive)
        if path is None:
            return None
        member = self.member_template.format(doc_id=doc_id, src=_normalise_src(src))
        found = self.index(path).get(member)
        if found is None:
            return None
        offset, size = found
        # Offset and size make this a range read rather than a tar extract, which is
        # the fastest of the three paths materialization dispatches to.
        return AssetLocator(path=path, member=member, byte_offset=offset, byte_size=size)


class ParquetIndexAssetResolver(AssetResolver):
    """The same mapping, precomputed: a parquet sidecar with one row per member.

    For runs that would rather not pay the header scan in every worker, and for archives
    that are expensive to walk. Member names must be unique across the index — the
    archive each member lives in comes from the index's own path column, not from the
    document's ``archive`` value, which is therefore ignored.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        index_path: str,
        member_column: str = "member",
        path_column: str = "path",
        offset_column: str = "byte_offset",
        size_column: str = "byte_size",
        member_template: str = "{doc_id}/{src}",
    ):
        """
        Args:
            index_path: Parquet file or directory of parquet files.
            member_column: Column holding the member name.
            path_column: Column holding the archive path.
            offset_column: Column holding the member's offset in the archive.
            size_column: Column holding the member's size in bytes.
            member_template: How a member is named, as in :class:`TarAssetResolver`.
        """
        self.index_path = index_path
        self.member_column = member_column
        self.path_column = path_column
        self.offset_column = offset_column
        self.size_column = size_column
        self.member_template = member_template
        self._index: dict[str, AssetLocator] | None = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_index"] = None
        return state

    def index(self) -> dict[str, AssetLocator]:
        """``{member: locator}``, read from the sidecar once per worker."""
        if self._index is None:
            import pyarrow.parquet as pq

            columns = [self.member_column, self.path_column, self.offset_column, self.size_column]
            # Loud, not degraded: a mis-named column is a configuration mistake, and
            # answering `None` to every row would report it as millions of missing
            # assets. Only an asset that is genuinely absent degrades a row.
            table = pq.read_table(self.index_path, columns=columns)
            self._index = {
                str(row[self.member_column]): AssetLocator(
                    path=str(row[self.path_column]),
                    member=str(row[self.member_column]),
                    byte_offset=None if row[self.offset_column] is None else int(row[self.offset_column]),
                    byte_size=None if row[self.size_column] is None else int(row[self.size_column]),
                )
                for row in table.to_pylist()
            }
            logger.debug(f"Loaded {len(self._index)} asset index entries from {self.index_path}")
        return self._index

    def resolve(self, doc_id: str, src: str, archive: str | None = None) -> AssetLocator | None:  # noqa: ARG002
        return self.index().get(self.member_template.format(doc_id=doc_id, src=_normalise_src(src)))


__all__ = [
    "ASSET_ATTRIBUTES",
    "AssetLocator",
    "AssetResolver",
    "ParquetIndexAssetResolver",
    "TarAssetResolver",
    "is_absolute_src",
    "split_multi_image_figures",
]
