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

"""Assets: every image must survive the converter, and say where its bytes are."""

from __future__ import annotations

import io
import pickle
import re
import tarfile
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from lxml import html as lxml_html

from nemo_curator.stages.text.html_extraction.assets import (
    AssetLocator,
    AssetResolver,
    ParquetIndexAssetResolver,
    TarAssetResolver,
    split_multi_image_figures,
)
from nemo_curator.stages.text.html_extraction.mineru_html import MinerUHtmlExtractStage
from nemo_curator.stages.text.html_extraction.mineru_interleaved import MinerUHtmlInterleavedStage
from nemo_curator.tasks import DocumentBatch, InterleavedBatch

if TYPE_CHECKING:
    from pathlib import Path

# The shape LaTeXML actually emits: several images inside one <span> inside one <p>,
# and a single <figcaption> for the lot.
THREE_IN_ONE_FIGURE = (
    "<div><p>Before.</p>"
    '<figure class="ltx_figure" id="S2.F1">'
    '<p class="ltx_p"><span class="ltx_text">'
    '<img src="x1.png" id="S2.F1.g1" class="ltx_graphics" alt="Refer to caption">'
    '<img src="x2.png" id="S2.F1.g2" class="ltx_graphics" alt="Refer to caption">'
    '<img src="x3.png" id="S2.F1.g3" class="ltx_graphics" alt="Refer to caption">'
    "</span></p>"
    "<figcaption>Figure 1: three panels.</figcaption>"
    "</figure>"
    "<p>After.</p></div>"
)

ONE_IN_ONE_FIGURE = (
    "<div><figure class=\"ltx_figure\"><p class=\"ltx_p\"><img src='x1.png' alt='a'></p>"
    "<figcaption>Figure 1: one panel.</figcaption></figure></div>"
)


def images(html: str) -> list[str]:
    return re.findall(r"<img[^>]*\bsrc=[\"']([^\"']+)", html)


def figures(html: str) -> list[lxml_html.HtmlElement]:
    return lxml_html.fromstring(html).findall(".//figure")


class TestSplitMultiImageFigures:
    """The transform is a workaround for a converter that keeps one image per figure.

    It has to earn its place twice: recover the images that would be dropped, and cost
    nothing — not even a re-serialisation — on the documents that never had the problem.
    """

    def test_each_image_gets_its_own_figure(self) -> None:
        out = split_multi_image_figures(THREE_IN_ONE_FIGURE)
        assert images(out) == ["x1.png", "x2.png", "x3.png"]
        assert [len(f.findall(".//img")) for f in figures(out)] == [1, 1, 1]

    def test_the_caption_is_not_duplicated(self) -> None:
        # One caption for three panels is what the document said. Copying it onto each
        # new figure would invent two captions the author never wrote.
        out = split_multi_image_figures(THREE_IN_ONE_FIGURE)
        assert out.count("Figure 1: three panels.") == 1
        captioned = [f for f in figures(out) if f.findall(".//figcaption")]
        assert len(captioned) == 1
        # ... and it stays with the last image, so the rows read image, image, caption.
        assert images(lxml_html.tostring(captioned[0], encoding="unicode")) == ["x3.png"]

    def test_the_image_keeps_the_markup_it_sat_in(self) -> None:
        # The converter's behaviour depends on what an image sits in, so the lifted
        # image arrives in a copy of its wrapping rather than bare under the figure.
        first = figures(split_multi_image_figures(THREE_IN_ONE_FIGURE))[0]
        assert first.find("p") is not None
        assert first.find("p/span") is not None
        assert first.find("p/span/img").get("src") == "x1.png"

    def test_the_duplicated_shells_do_not_duplicate_ids(self) -> None:
        # Two elements with id="S2.F1" is invalid, and would confuse anything that
        # later indexes the document by id. The images keep their own ids: they moved,
        # they were not copied.
        out = split_multi_image_figures(THREE_IN_ONE_FIGURE)
        ids = re.findall(r'id="([^"]+)"', out)
        assert sorted(ids) == sorted(set(ids))
        assert {"S2.F1", "S2.F1.g1", "S2.F1.g2", "S2.F1.g3"} == set(ids)

    def test_a_single_image_figure_is_returned_untouched(self) -> None:
        # Byte-identical, not merely equivalent: most documents look like this, and a
        # re-parse that "only" normalises quoting still rewrites every document in the
        # corpus for no reason.
        assert split_multi_image_figures(ONE_IN_ONE_FIGURE) is ONE_IN_ONE_FIGURE

    @pytest.mark.parametrize(
        "html",
        [
            "",
            "<div><p>No pictures here.</p></div>",
            '<div><img src="a.png"><img src="b.png"></div>',  # images, but no figure
        ],
    )
    def test_nothing_to_split_changes_nothing(self, html: str) -> None:
        assert split_multi_image_figures(html) is html

    def test_sub_figures_are_left_alone(self) -> None:
        # Sub-figures are nested figures that already own one image each. The converter
        # handles those; splitting them would be re-shaping markup that works.
        nested = (
            '<div><figure class="ltx_figure">'
            '<figure class="ltx_subfigure"><img src="x1.png"><figcaption>(a)</figcaption></figure>'
            '<figure class="ltx_subfigure"><img src="x2.png"><figcaption>(b)</figcaption></figure>'
            "<figcaption>Figure 1: two panels.</figcaption></figure></div>"
        )
        assert split_multi_image_figures(nested) is nested

    def test_no_prose_is_lost(self) -> None:
        # lxml hangs the text *after* a node off that node, so lifting an image the
        # naive way deletes the words that followed it — invisibly, since the document
        # still reads as prose, just shorter.
        html = (
            '<div><figure><p>left <img src="x1.png"> middle <img src="x2.png"> right</p>'
            "<figcaption>Cap.</figcaption></figure></div>"
        )
        out = split_multi_image_figures(html)
        assert lxml_html.fromstring(out).text_content() == lxml_html.fromstring(html).text_content()

    def test_a_figure_that_is_the_whole_fragment_is_left_whole(self) -> None:
        # It has nothing to become a sibling of, and lxml keeps the wrapper it parsed
        # into outside the root it returns — so a figure inserted beside this one is
        # silently dropped on the way out, taking its image with it. Leaving the
        # fragment alone loses nothing; splitting it lost the first image.
        root = '<figure><img src="x1.png"><img src="x2.png"></figure>'
        assert split_multi_image_figures(root) is root


class TestExtractStageWiring:
    """The transform has to run on the way into the converter, and nowhere else."""

    def _rendered_html(self, **kwargs) -> str:
        stage = MinerUHtmlExtractStage(**kwargs)
        seen: list[str] = []

        def capture(main_html: str, url: str | None, output_format: str) -> str:  # noqa: ARG001
            seen.append(main_html)
            return "md"

        stage._convert = capture
        stage._render([THREE_IN_ONE_FIGURE], [None], ["ok"])
        return seen[0]

    def test_figures_are_split_before_conversion(self) -> None:
        assert len(figures(self._rendered_html())) == 3

    def test_the_workaround_can_be_turned_off(self) -> None:
        assert self._rendered_html(split_figures=False) is THREE_IN_ONE_FIGURE


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A tar shaped like the asset shards: members named `{doc_id}/{filename}`."""
    path = tmp_path / "assets" / "shard-a" / "part-0000.tar"
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w") as tar:
        for name, payload in (
            ("astro-ph/0001020/x1.png", b"\x89PNG-one" * 40),
            ("astro-ph/0001020/x2.png", b"\x89PNG-two" * 7),
            ("astro-ph/0001031/x1.png", b"\x89PNG-three" * 3),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    return path


class TestTarAssetResolver:
    def test_a_member_is_found_where_the_locator_says_it_is(self, archive: Path) -> None:
        # The point of recording an offset and a size is that a reader can seek
        # straight to the bytes without opening the tar. If they are wrong, nothing
        # fails — it reads the wrong bytes — so the test reads them.
        resolver = TarAssetResolver(archive=str(archive))
        located = resolver.resolve("astro-ph/0001020", "x2.png")

        assert located == AssetLocator(
            path=str(archive),
            member="astro-ph/0001020/x2.png",
            byte_offset=located.byte_offset,
            byte_size=len(b"\x89PNG-two" * 7),
        )
        with archive.open("rb") as handle:
            handle.seek(located.byte_offset)
            assert handle.read(located.byte_size) == b"\x89PNG-two" * 7

    def test_an_asset_that_is_not_there_resolves_to_nothing(self, archive: Path) -> None:
        resolver = TarAssetResolver(archive=str(archive))
        assert resolver.resolve("astro-ph/0001020", "x9.png") is None
        assert resolver.resolve("astro-ph/9999999", "x1.png") is None

    def test_the_headers_are_read_once_per_archive(self, archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Walking a 500 MB shard's headers per image instead of per shard is the
        # difference between a stage that runs and one that does not.
        opens = []
        real_open = tarfile.open
        monkeypatch.setattr(tarfile, "open", lambda *a, **k: (opens.append(a), real_open(*a, **k))[1])

        resolver = TarAssetResolver(archive=str(archive))
        for _ in range(3):
            resolver.resolve("astro-ph/0001020", "x1.png")
        assert len(opens) == 1

    def test_an_unreadable_archive_degrades_and_says_so_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken = tmp_path / "not-a.tar"
        broken.write_bytes(b"this is not a tar file")
        opens = []
        real_open = tarfile.open
        monkeypatch.setattr(tarfile, "open", lambda *a, **k: (opens.append(a), real_open(*a, **k))[1])

        resolver = TarAssetResolver(archive=str(broken))
        assert resolver.resolve("d", "x1.png") is None
        assert resolver.resolve("d", "x2.png") is None
        assert len(opens) == 1  # the failure is remembered, not retried per image

    def test_the_archive_template_chooses_the_tar_for_a_shard(self, archive: Path) -> None:
        # This is the only place a corpus's layout is allowed to live: configuration
        # the driver passes, not knowledge the stage has.
        resolver = TarAssetResolver(
            archive_template=str(archive.parent.parent / "{archive}" / "part-0000.tar"),
        )
        assert resolver.resolve("astro-ph/0001020", "x1.png", "shard-a") is not None
        assert resolver.resolve("astro-ph/0001020", "x1.png", "shard-b") is None
        assert resolver.resolve("astro-ph/0001020", "x1.png", None) is None

    def test_a_relative_src_is_normalised(self, archive: Path) -> None:
        resolver = TarAssetResolver(archive=str(archive))
        assert resolver.resolve("astro-ph/0001020", "./x1.png") is not None
        assert resolver.resolve("astro-ph/0001020", "x1.png?v=2") is not None

    def test_the_index_does_not_travel_to_the_workers(self, archive: Path) -> None:
        # A stage is pickled to every worker. Shipping a shard's index with it would
        # pay for the scan once per worker and once more per pickle.
        resolver = TarAssetResolver(archive=str(archive))
        resolver.resolve("astro-ph/0001020", "x1.png")
        assert resolver._indices

        revived = pickle.loads(pickle.dumps(resolver))  # noqa: S301
        assert not revived._indices
        assert revived.resolve("astro-ph/0001020", "x1.png") == resolver.resolve("astro-ph/0001020", "x1.png")


class TestParquetIndexAssetResolver:
    @pytest.fixture
    def index(self, archive: Path, tmp_path: Path) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tarfile.open(archive) as tar:
            rows = [
                {
                    "member": member.name,
                    "path": str(archive),
                    "byte_offset": member.offset_data,
                    "byte_size": member.size,
                }
                for member in tar
                if member.isfile()
            ]
        path = tmp_path / "asset-index.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def test_it_answers_exactly_what_the_tar_scan_answers(self, index: Path, archive: Path) -> None:
        # Two ways of learning the same thing. If they disagree, one of them is wrong
        # and rows written by different runs of the same pipeline would not match.
        precomputed = ParquetIndexAssetResolver(str(index))
        scanned = TarAssetResolver(archive=str(archive))
        for doc_id, src in (("astro-ph/0001020", "x1.png"), ("astro-ph/0001031", "x1.png")):
            assert precomputed.resolve(doc_id, src) == scanned.resolve(doc_id, src)

    def test_an_asset_that_is_not_indexed_resolves_to_nothing(self, index: Path) -> None:
        assert ParquetIndexAssetResolver(str(index)).resolve("astro-ph/0001020", "x9.png") is None

    def test_a_mis_named_column_fails_loudly(self, index: Path) -> None:
        # Answering `None` would report a configuration mistake as millions of missing
        # assets. Only an asset that is genuinely absent is allowed to degrade a row.
        with pytest.raises(Exception, match=r"(?i)offsets"):
            ParquetIndexAssetResolver(str(index), offset_column="offsets").resolve("astro-ph/0001020", "x1.png")

    def test_the_index_does_not_travel_to_the_workers(self, index: Path) -> None:
        resolver = ParquetIndexAssetResolver(str(index))
        resolver.resolve("astro-ph/0001020", "x1.png")
        assert pickle.loads(pickle.dumps(resolver))._index is None  # noqa: S301


class ExplodingResolver(AssetResolver):
    """A resolver that must not be asked."""

    def resolve(self, doc_id: str, src: str, archive: str | None = None) -> AssetLocator | None:
        msg = f"resolver consulted for {src!r}"
        raise AssertionError(msg)


DOC_ID = "astro-ph/0001020"
DOC_URL = "https://arxiv.org/abs/astro-ph/0001020"


def rows_for(markdown: str, archive: Path | None = None, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("assets", TarAssetResolver(archive=str(archive)) if archive is not None else None)
    stage = MinerUHtmlInterleavedStage(id_field="arxiv_id", url_field="url", **kwargs)
    batch = DocumentBatch(
        dataset_name="t",
        data=pd.DataFrame([{"arxiv_id": DOC_ID, "url": DOC_URL, "text": markdown}]),
    )
    return stage.process(batch).to_pandas()


class TestInterleavedAssetResolution:
    def test_a_relative_src_becomes_an_archive_locator(self, archive: Path) -> None:
        out = rows_for("Intro.\n\n![](x1.png)\n\nFigure 1: cap.", archive)
        image = out[out.modality == "image"].iloc[0]

        located = InterleavedBatch.parse_source_ref(image["source_ref"])
        assert located["path"] == str(archive)
        assert located["member"] == "astro-ph/0001020/x1.png"
        assert located["byte_size"] == len(b"\x89PNG-one" * 40)
        with archive.open("rb") as handle:
            handle.seek(located["byte_offset"])
            assert handle.read(located["byte_size"]) == b"\x89PNG-one" * 40
        assert image["materialize_error"] is None
        assert image["content_type"] == "image/png"

    def test_the_row_order_is_still_document_order(self, archive: Path) -> None:
        # Two images then their caption. The association is carried by order, and
        # nothing here regroups or pairs them.
        out = rows_for("![](x1.png)\n\n![](x2.png)\n\nFigure 1: cap.", archive)
        assert list(out["modality"]) == ["image", "image", "text"]
        assert list(out["position"]) == [0, 1, 2]

    def test_a_missing_asset_degrades_its_own_row(self, archive: Path) -> None:
        # A corpus where some figures were never shipped is normal. A document lost
        # because one of them was is not: the row keeps its place, and says why it is
        # empty in the column that exists for exactly that.
        out = rows_for("![](x1.png)\n\n![](gone.png)\n\nTail.", archive)
        assert list(out["modality"]) == ["image", "image", "text"]

        missing = out.iloc[1]
        assert missing["source_ref"] is None
        assert "gone.png" in missing["materialize_error"]
        assert out.iloc[0]["source_ref"] is not None
        assert out.iloc[0]["materialize_error"] is None

    def test_an_absolute_src_is_recorded_as_it_stands(self) -> None:
        # It is already an address; there is no archive member to look for.
        out = rows_for("![](https://cdn/x1.png)", assets=ExplodingResolver())
        assert InterleavedBatch.parse_source_ref(out.iloc[0]["source_ref"])["path"] == "https://cdn/x1.png"

    def test_without_a_resolver_the_src_is_recorded_as_written(self) -> None:
        out = rows_for("![](x1.png)")
        located = InterleavedBatch.parse_source_ref(out.iloc[0]["source_ref"])
        assert (located["path"], located["member"]) == ("x1.png", None)

    def test_the_archive_column_picks_the_shard(self, archive: Path) -> None:
        stage = MinerUHtmlInterleavedStage(
            id_field="arxiv_id",
            asset_archive_field="shard",
            assets=TarAssetResolver(archive_template=str(archive.parent.parent / "{archive}" / "part-0000.tar")),
        )
        batch = DocumentBatch(
            dataset_name="t",
            data=pd.DataFrame(
                [
                    {"arxiv_id": "astro-ph/0001020", "shard": "shard-a", "text": "![](x1.png)"},
                    {"arxiv_id": "astro-ph/0001020", "shard": "shard-z", "text": "![](x1.png)"},
                ]
            ),
        )
        out = stage.process(batch).to_pandas()
        assert out.iloc[0]["source_ref"] is not None
        assert out.iloc[1]["source_ref"] is None

    def test_the_document_id_can_be_a_different_column_from_the_sample_id(self, archive: Path) -> None:
        stage = MinerUHtmlInterleavedStage(
            url_field="url",
            asset_id_field="arxiv_id",
            assets=TarAssetResolver(archive=str(archive)),
        )
        batch = DocumentBatch(
            dataset_name="t",
            data=pd.DataFrame([{"arxiv_id": DOC_ID, "url": DOC_URL, "text": "![](x1.png)"}]),
        )
        out = stage.process(batch).to_pandas()
        assert out.iloc[0]["sample_id"] == DOC_URL
        assert InterleavedBatch.parse_source_ref(out.iloc[0]["source_ref"])["member"] == "astro-ph/0001020/x1.png"

    def test_a_resolver_that_is_never_consulted_is_visible(self, archive: Path) -> None:
        # The misconfiguration that looks like success: forget `url_field=None` on the
        # extract stage and every src arrives absolute, so every row gets a locator, the
        # resolver is never asked, and the locators are URLs the converter invented.
        # Counted separately so the run's metrics can say so.
        stage = MinerUHtmlInterleavedStage(id_field="arxiv_id", assets=TarAssetResolver(archive=str(archive)))
        batch = DocumentBatch(
            dataset_name="t",
            data=pd.DataFrame([{"arxiv_id": DOC_ID, "text": "![](https://arxiv.org/abs/x1.png)"}]),
        )
        stage.process(batch)
        assert stage._custom_metrics == {
            "assets_resolved": 0,
            "assets_unresolved": 0,
            "assets_already_absolute": 1,
        }

    def test_the_metrics_count_what_was_found_and_what_was_not(self, archive: Path) -> None:
        stage = MinerUHtmlInterleavedStage(id_field="arxiv_id", assets=TarAssetResolver(archive=str(archive)))
        batch = DocumentBatch(
            dataset_name="t",
            data=pd.DataFrame([{"arxiv_id": DOC_ID, "text": "![](x1.png)\n\n![](x2.png)\n\n![](gone.png)"}]),
        )
        stage.process(batch)
        assert stage._custom_metrics == {
            "assets_resolved": 2,
            "assets_unresolved": 1,
            "assets_already_absolute": 0,
        }

    def test_text_rows_are_untouched_by_any_of_this(self, archive: Path) -> None:
        out = rows_for("Intro.\n\n![](x1.png)\n\nTail.", archive)
        text_rows = out[out.modality == "text"]
        assert list(text_rows["text_content"]) == ["Intro.", "Tail."]
        assert text_rows["source_ref"].isna().all()
        assert text_rows["materialize_error"].isna().all()
