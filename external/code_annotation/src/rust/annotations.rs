// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Core annotation functions for code analysis.

use bpe;
use bpe_openai;
use flate2::read::GzDecoder;
use hyperpolyglot;
use lazy_regex::Regex;
use lazy_static::lazy_static;
use polars::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyErr;
use pyo3_polars::PyDataFrame;
#[cfg(feature = "software_metrics")]
use software_metrics::{get_metrics, has_parsing_errors};
use std::any::Any;
use std::borrow::Cow;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Cursor, Read};
use std::str;
use std::sync::Mutex;
use tiktoken_rs;

mod vendored_loc;

const CODE_COLUMN_NAME: &str = "content";
const LANGUAGE_COL_NAME: &str = "language";
const TOKENS_COL_NAME: &str = "tokenized_content";
const FILENAME_COLUMN_NAME: &str = "representative_filename";
const COMPRESSED_SRC_COLUMN_NAME: &str = "compressed_content";
const REQUIRED_COLUMNS: [&str; 1] = [COMPRESSED_SRC_COLUMN_NAME];

const TIKTOKEN_O200K_BASE_NAME: &str = "tiktoken_o200k_base";
const GITHUB_O200K_BASE_NAME: &str = "github_o200k_base";

use macros::{register, FromHashMap};

// Type definitions for function registry
type FunctionsArgs<'a> = HashMap<String, Bound<'a, PyAny>>;
type Callable = Box<dyn Fn(FunctionsArgs, &mut DataFrame) -> PyResult<()> + Send>;
type TypeChecker = Box<dyn Fn(&FunctionsArgs) -> Result<Box<dyn Any>, PyErr> + Send>;

lazy_static! {
    pub static ref FN_REGISTRY: Mutex<HashMap<String, Callable>> = Mutex::new(HashMap::new());
    pub static ref INPUT_TYPE_REGISTRY: Mutex<HashMap<String, TypeChecker>> =
        Mutex::new(HashMap::new());
}

/// Get the column name from an annotation function.
macro_rules! get_col_name {
    () => {{
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        fn f() {}
        let name = type_name_of(f);
        let parts: Vec<&str> = name.split("::").collect();
        if parts.len() < 2 {
            panic!("Problems with function name extraction. Full name: {}\n", name)
        }
        format!("{}", parts[parts.len() - 2])
    }};
}

/// Parse function names/args and call registered functions.
pub fn call_functions(
    function_names: Vec<String>,
    function_args: Vec<HashMap<String, Bound<'_, PyAny>>>,
    df_ref: &mut DataFrame,
) -> PyResult<PyDataFrame> {
    let fn_registry = FN_REGISTRY.lock().unwrap();
    let input_type_registry = INPUT_TYPE_REGISTRY.lock().unwrap();
    let mut functions = Vec::new();

    // Check required columns
    let df_colnames = df_ref.get_column_names_str();
    let missing_required_cols = REQUIRED_COLUMNS
        .iter()
        .filter(|colname| !df_colnames.contains(colname))
        .copied()
        .collect::<Vec<&str>>();
    if !missing_required_cols.is_empty() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Missing required columns: '{}'.",
            missing_required_cols.join(", ")
        )));
    }

    // Validate functions and arguments
    for (func_name, args) in function_names.iter().zip(function_args.clone()) {
        let fn_name_str = func_name.as_str();
        if !fn_registry.contains_key(fn_name_str) {
            return Err(PyErr::new::<PyTypeError, _>(format!(
                "Function '{}' does not exist.",
                func_name
            )));
        }
        let convert = input_type_registry.get(fn_name_str).unwrap();
        convert(&args)?;
        functions.push(fn_registry.get(fn_name_str).unwrap());
    }

    // Decompress code column
    let mut df = std::mem::take(df_ref)
        .lazy()
        .with_column(
            col(COMPRESSED_SRC_COLUMN_NAME)
                .map(
                    |s| {
                        let decompressed_strings: Vec<Option<String>> = s
                            .binary()?
                            .iter()
                            .map(|opt_bytes| {
                                opt_bytes.and_then(|bytes| {
                                    let mut decoder = GzDecoder::new(Cursor::new(bytes));
                                    let mut decompressed_data = String::new();
                                    match decoder.read_to_string(&mut decompressed_data) {
                                        Ok(_) => Some(decompressed_data),
                                        Err(_) => None,
                                    }
                                })
                            })
                            .collect();
                        let series = Series::new("decompressed".into(), decompressed_strings);
                        Ok(Some(series.into()))
                    },
                    GetOutput::from_type(DataType::String),
                )
                .alias(CODE_COLUMN_NAME),
        )
        .collect()
        .unwrap();

    // Call each function
    let mut n_cols = df.width();
    for ((func, func_name), args) in functions.iter().zip(function_names).zip(function_args) {
        func(args, &mut df)?;
        if df.width() == n_cols {
            panic!(
                "Column count unchanged after calling {}. Use get_col_name!() macro.",
                func_name
            );
        }
        n_cols = df.width();
    }

    Ok(PyDataFrame(df))
}

// ============================================================================
// Argument Structs
// ============================================================================

#[derive(FromHashMap)]
struct BasicArgs {
    xml_header_search_length: usize,
    max_decompressed_byte_size: Option<usize>,
}

#[derive(FromHashMap)]
struct DetectLanguageArgs {}

#[cfg(feature = "software_metrics")]
#[derive(FromHashMap)]
struct SoftwareMetricsArgs {}

#[derive(FromHashMap)]
struct TokenizeArgs {
    tokenizer_name: String,
    vocab: Option<String>,
    pretokenizer_patterns: Option<Vec<(String, bool)>>,
}

#[derive(FromHashMap)]
struct OpenCoderRsArgs {}

#[derive(FromHashMap)]
struct DecontaminateArgs {
    ngrams: HashMap<String, Vec<String>>,
    ngram_order: usize,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_code_col(df: &DataFrame) -> &ChunkedArray<StringType> {
    df.column(CODE_COLUMN_NAME)
        .unwrap()
        .str()
        .expect("Problem converting code column to strings.")
}

fn extract_all_patterns(df: &mut DataFrame, annotation_prefix: &str) {
    let base64_pattern = Regex::new(r"[a-zA-Z0-9+/\n=]{64,}").unwrap();
    let hex_pattern = Regex::new(r"(?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}").unwrap();
    let unicode_pattern = Regex::new(r"(?:\\u[0-9a-fA-F]{4}){8,}").unwrap();

    let dflen = df.height();
    let mut alpha_vec: Vec<f32> = vec![0.0; dflen];
    let mut alphanum_vec: Vec<f32> = vec![0.0; dflen];
    let mut base64_vec: Vec<f32> = vec![0.0; dflen];
    let mut hex_vec: Vec<f32> = vec![0.0; dflen];
    let mut unicode_vec: Vec<f32> = vec![0.0; dflen];

    for (idx, opt_s) in get_code_col(df).iter().enumerate() {
        if let Some(s) = opt_s {
            let slen = s.len() as f32;
            alpha_vec[idx] = (s.chars().filter(|c| c.is_alphabetic()).count() as f32) / slen;
            alphanum_vec[idx] = s.chars().filter(|c| c.is_alphanumeric()).count() as f32 / slen;
            base64_vec[idx] =
                base64_pattern.find_iter(s).map(|m| m.len()).sum::<usize>() as f32 / slen;
            hex_vec[idx] =
                hex_pattern.find_iter(s).map(|m| m.len()).sum::<usize>() as f32 / slen;
            unicode_vec[idx] =
                unicode_pattern.find_iter(s).map(|m| m.len()).sum::<usize>() as f32 / slen;
        }
    }

    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_alpha_percent").into(),
        alpha_vec,
    ))
    .unwrap();

    df.with_column(Series::new(
        format!("{annotation_prefix}_alnum_percent").into(),
        alphanum_vec,
    ))
    .unwrap();

    df.with_column(Series::from_vec(
        format!("{annotation_prefix}_base64_percent").into(),
        base64_vec,
    ))
    .unwrap();

    df.with_column(Series::from_vec(
        format!("{annotation_prefix}_hex_percent").into(),
        hex_vec,
    ))
    .unwrap();

    df.with_column(Series::from_vec(
        format!("{annotation_prefix}_unicode_percent").into(),
        unicode_vec,
    ))
    .unwrap();

    // Pattern match lengths
    let code_col = get_code_col(df);
    df.with_column(code_pattern_match_lens(
        code_col,
        &base64_pattern,
        format!("{annotation_prefix}_base64_match_lengths"),
    ))
    .unwrap();

    let code_col = get_code_col(df);
    df.with_column(code_pattern_match_lens(
        code_col,
        &hex_pattern,
        format!("{annotation_prefix}_hex_match_lengths"),
    ))
    .unwrap();

    let code_col = get_code_col(df);
    df.with_column(code_pattern_match_lens(
        code_col,
        &unicode_pattern,
        format!("{annotation_prefix}_unicode_match_lengths"),
    ))
    .unwrap();
}

fn code_pattern_match_lens(
    code_col: &ChunkedArray<StringType>,
    pattern: &Regex,
    col_name: String,
) -> Series {
    Series::new(
        col_name.into(),
        code_col
            .iter()
            .map(|opt_text| {
                opt_text.map(|s| {
                    pattern
                        .find_iter(s)
                        .map(|m| m.len() as u64)
                        .collect::<Series>()
                })
            })
            .collect::<ChunkedArray<ListType>>(),
    )
}

// ============================================================================
// Registered Functions
// ============================================================================

#[register]
fn detect_language(_: DetectLanguageArgs, df: &mut DataFrame) -> PyResult<()> {
    let (languages, variants): (Vec<Option<String>>, Vec<Option<String>>) = get_code_col(df)
        .iter()
        .zip(
            df.column(FILENAME_COLUMN_NAME)
                .unwrap()
                .str()
                .unwrap(),
        )
        .map(|(code_opt, fname_opt)| {
            if let Some(code) = code_opt {
                match language(fname_opt, code) {
                    Ok(Some(detection)) => (
                        Some(detection.language().to_string()),
                        Some(detection.variant().to_string()),
                    ),
                    _ => (None, None),
                }
            } else {
                (None, None)
            }
        })
        .unzip();

    df.with_column(Series::new(LANGUAGE_COL_NAME.into(), languages))
        .unwrap()
        .with_column(Series::new("language_detector".into(), variants))
        .unwrap();
    Ok(())
}

#[register]
fn basic(args: BasicArgs, df: &mut DataFrame) -> PyResult<()> {
    let annotation_prefix = get_col_name!();

    let num_bytes_colname = format!("{annotation_prefix}_num_bytes");
    df.with_column::<ChunkedArray<UInt32Type>>(
        get_code_col(df)
            .str_len_bytes()
            .with_name(num_bytes_colname.into()),
    )
    .unwrap();

    df.with_column(Series::new(
        format!("{annotation_prefix}_valid_utf8").into(),
        get_code_col(df)
            .iter()
            .map(|s| s.is_some())
            .collect::<Vec<bool>>(),
    ))
    .unwrap();

    let byte_size_filter_threshold = args.max_decompressed_byte_size.unwrap_or(usize::MAX);
    df.with_column(get_code_col(df).apply(|opt_code| match opt_code {
        Some(code) if code.len() > byte_size_filter_threshold => None,
        Some(code) => Some(Cow::Borrowed(code)),
        None => None,
    }))
    .unwrap();

    df.with_column::<ChunkedArray<UInt64Type>>(
        get_code_col(df)
            .apply_nonnull_values_generic(DataType::UInt64, |s| {
                s.lines().map(|line| line.len()).max().unwrap_or(0) as u64
            })
            .with_name(format!("{annotation_prefix}_max_line_length").into()),
    )
    .unwrap();

    extract_all_patterns(df, &annotation_prefix);

    df.with_column::<ChunkedArray<UInt64Type>>(
        get_code_col(df)
            .apply_nonnull_values_generic(DataType::UInt64, |s| s.lines().count() as u64)
            .with_name(format!("{annotation_prefix}_num_lines").into()),
    )
    .unwrap();

    df.with_column::<ChunkedArray<Float32Type>>(
        get_code_col(df)
            .apply_nonnull_values_generic(DataType::Float32, |s| {
                s.len() as f32 / s.lines().count() as f32
            })
            .with_name(format!("{annotation_prefix}_average_line_length").into()),
    )
    .unwrap();

    df.with_column::<ChunkedArray<BooleanType>>(
        get_code_col(df)
            .apply_nonnull_values_generic(DataType::Boolean, |s| {
                let mut search_end = min(args.xml_header_search_length, s.len());
                while !s.is_char_boundary(search_end) {
                    search_end += 1;
                }
                s[0..search_end].contains("<?xml version=")
            })
            .with_name(format!("{annotation_prefix}_contains_xml_header").into()),
    )
    .unwrap();

    Ok(())
}

/// Detect language with hyperpolyglot.
pub fn language(
    filename: Option<&str>,
    content: &str,
) -> Result<Option<hyperpolyglot::Detection>, std::io::Error> {
    let candidate = filename
        .and_then(|filename| hyperpolyglot::detectors::get_language_from_filename(filename));
    if let Some(candidate) = candidate {
        return Ok(Some(hyperpolyglot::Detection::Filename(candidate)));
    };

    let extension = filename.and_then(|filename| hyperpolyglot::detectors::get_extension(filename));
    let candidates = extension
        .map(|ext| hyperpolyglot::detectors::get_languages_from_extension(ext))
        .unwrap_or_else(Vec::new);

    if candidates.len() == 1 {
        return Ok(Some(hyperpolyglot::Detection::Extension(candidates[0])));
    };

    let mut reader = BufReader::new(content.as_bytes());
    let candidates = hyperpolyglot::filter_candidates(
        candidates,
        hyperpolyglot::detectors::get_languages_from_shebang(&mut reader)?,
    );
    if candidates.len() == 1 {
        return Ok(Some(hyperpolyglot::Detection::Shebang(candidates[0])));
    };

    let content = hyperpolyglot::truncate(content);
    let candidates = if candidates.len() > 1 {
        if let Some(extension) = extension {
            let languages = hyperpolyglot::detectors::get_languages_from_heuristics(
                &extension[..],
                &candidates,
                &content,
            );
            hyperpolyglot::filter_candidates(candidates, languages)
        } else {
            candidates
        }
    } else {
        candidates
    };

    match candidates.len() {
        1 => Ok(Some(hyperpolyglot::Detection::Heuristics(candidates[0]))),
        _ => Ok(Some(hyperpolyglot::Detection::Classifier(
            hyperpolyglot::detectors::classify(&content, &candidates),
        ))),
    }
}

#[cfg(feature = "software_metrics")]
#[register]
fn software_metrics(_: SoftwareMetricsArgs, df: &mut DataFrame) -> PyResult<()> {
    let annotation_prefix = get_col_name!();
    let dflen = df.height();

    let mut cyclomatic: Vec<f32> = vec![0.0; dflen];
    let mut cognitive_average: Vec<f32> = vec![0.0; dflen];
    let mut exits_average: Vec<f32> = vec![0.0; dflen];
    let mut maintainability_index: Vec<f32> = vec![0.0; dflen];
    let mut halstead_difficulty: Vec<f32> = vec![0.0; dflen];
    let mut comment_lines: Vec<f32> = vec![0.0; dflen];
    let mut comment_lines_frac: Vec<f32> = vec![0.0; dflen];
    let mut comments_per_space: Vec<f32> = vec![0.0; dflen];
    let mut blank_lines: Vec<f32> = vec![0.0; dflen];
    let mut blank_lines_per_space: Vec<f32> = vec![0.0; dflen];
    let mut args_average: Vec<f32> = vec![0.0; dflen];
    let mut functions_closures_per_space: Vec<f32> = vec![0.0; dflen];
    let mut total_cda: Vec<f32> = vec![0.0; dflen];
    let mut total_wmc: Vec<f32> = vec![0.0; dflen];
    let mut total_coa: Vec<f32> = vec![0.0; dflen];
    let mut parsed_ok: Vec<bool> = vec![false; dflen];

    for ((idx, opt_code), opt_lang) in get_code_col(df).iter().enumerate().zip(
        df.column(LANGUAGE_COL_NAME).unwrap().str().unwrap(),
    ) {
        if let (Some(code), Some(lang)) = (opt_code, opt_lang) {
            parsed_ok[idx] = !has_parsing_errors(code, lang);
            if let Some(mzmetrics) = get_metrics(lang, code) {
                let metrics = mzmetrics.metrics;
                cyclomatic[idx] = metrics.cyclomatic.cyclomatic_average() as f32;
                cognitive_average[idx] = metrics.cognitive.cognitive_average() as f32;
                exits_average[idx] = metrics.nexits.exit_average() as f32;
                maintainability_index[idx] = metrics.mi.mi_visual_studio() as f32;
                halstead_difficulty[idx] = metrics.halstead.difficulty() as f32;
                comment_lines[idx] = metrics.loc.cloc() as f32;
                comment_lines_frac[idx] =
                    (metrics.loc.cloc() / (1. + metrics.loc.sloc())) as f32;
                comments_per_space[idx] = metrics.loc.cloc_average() as f32;
                blank_lines[idx] = metrics.loc.blank() as f32;
                blank_lines_per_space[idx] = metrics.loc.blank_average() as f32;
                args_average[idx] = metrics.nargs.nargs_average() as f32;
                functions_closures_per_space[idx] = metrics.nom.average() as f32;

                let cda = metrics.npa.total_cda();
                if !cda.is_nan() {
                    total_cda[idx] = cda as f32;
                    total_wmc[idx] = metrics.wmc.total_wmc() as f32;
                }
                let coa = metrics.npm.total_coa();
                if !coa.is_nan() {
                    total_coa[idx] = coa as f32;
                }
            }
        }
    }

    // Add all columns
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_cyclomatic_complexity").into(),
        cyclomatic,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_cognitive_complexity").into(),
        cognitive_average,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_exits_average").into(),
        exits_average,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_maintainability_index").into(),
        maintainability_index,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_halstead_difficulty").into(),
        halstead_difficulty,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_comment_lines").into(),
        comment_lines,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_comment_lines_frac").into(),
        comment_lines_frac,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_comment_lines_per_space").into(),
        comments_per_space,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_blank_lines").into(),
        blank_lines,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_blank_lines_per_space").into(),
        blank_lines_per_space,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_args_average").into(),
        args_average,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_functions_closures_per_space").into(),
        functions_closures_per_space,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_total_cda").into(),
        total_cda,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_total_wmc").into(),
        total_wmc,
    )).unwrap();
    df.with_column(ChunkedArray::<Float32Type>::from_vec(
        format!("{annotation_prefix}_total_coa").into(),
        total_coa,
    )).unwrap();
    df.with_column(Series::new(
        format!("{annotation_prefix}_parsed_ok").into(),
        ChunkedArray::<BooleanType>::from_iter(parsed_ok),
    )).unwrap();

    Ok(())
}

/// Create a bpe_openai tokenizer from a custom vocabulary.
fn get_tokenizer_from_vocab(
    vocab: &str,
    patterns: Option<Vec<(String, bool)>>,
) -> bpe_openai::Tokenizer {
    let bpe = bpe::byte_pair_encoding::BytePairEncoding::from_tiktoken(vocab, None)
        .expect("Failed loading tokenizer vocabulary.");
    match patterns {
        None => {
            let pat1 = [
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+",
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*",
                "\\p{N}",
                " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                "\\s*[\\r\\n]+",
                "\\s+$",
            ].join("|");
            let pat2 = "\\s+\\s";
            let pat3 = "\\s+";
            bpe_openai::Tokenizer::new_lookahead(
                bpe,
                &[(&pat1, false), (pat2, true), (pat3, false)],
            ).unwrap()
        }
        Some(pats) => bpe_openai::Tokenizer::new_lookahead(
            bpe,
            &pats.iter().map(|(pat, is_lookahead)| (pat.as_str(), *is_lookahead)).collect::<Vec<_>>(),
        ).unwrap(),
    }
}

#[register]
fn tokenize(args: TokenizeArgs, df: &mut DataFrame) -> PyResult<()> {
    if args.tokenizer_name == TIKTOKEN_O200K_BASE_NAME {
        let tok = tiktoken_rs::o200k_base().unwrap();
        df.with_column(Series::new(
            TOKENS_COL_NAME.into(),
            get_code_col(df)
                .iter()
                .map(|opt_text| {
                    opt_text.map(|s| {
                        tok.encode_with_special_tokens(s)
                            .into_iter()
                            .collect::<Series>()
                    })
                })
                .collect::<ChunkedArray<ListType>>(),
        )).unwrap();
    } else {
        let process_column = |tokenizer: &bpe_openai::Tokenizer, df: &mut DataFrame| {
            df.with_column(Series::new(
                TOKENS_COL_NAME.into(),
                get_code_col(df)
                    .iter()
                    .map(|opt_text| {
                        opt_text.map(|s| tokenizer.encode(s).into_iter().collect::<Series>())
                    })
                    .collect::<ChunkedArray<ListType>>(),
            )).unwrap();
        };

        if args.tokenizer_name == GITHUB_O200K_BASE_NAME {
            process_column(bpe_openai::o200k_base(), df);
        } else if let Some(vocab) = args.vocab {
            let tokenizer = get_tokenizer_from_vocab(&vocab, args.pretokenizer_patterns);
            process_column(&tokenizer, df);
        } else {
            panic!("Provide a vocabulary for custom tokenizer {}", args.tokenizer_name);
        }
    }

    let token_len_col_name = format!("num_tokens_{}", args.tokenizer_name);
    df.with_column(Series::new(
        token_len_col_name.into(),
        df.column(TOKENS_COL_NAME)
            .unwrap()
            .list()
            .unwrap()
            .iter()
            .map(|opt_vec| opt_vec.map(|vec| vec.len() as u64))
            .collect::<ChunkedArray<UInt64Type>>(),
    )).unwrap();

    Ok(())
}

#[register]
fn opencoder_rs(_: OpenCoderRsArgs, df: &mut DataFrame) -> PyResult<()> {
    let mut comment_lines_fracs = Vec::<Option<f32>>::with_capacity(df.height());
    let mut comment_chars_fracs = Vec::<Option<f32>>::with_capacity(df.height());

    for (opt_code, opt_lang) in get_code_col(df).iter().zip(
        df.column(LANGUAGE_COL_NAME).unwrap().str().unwrap(),
    ) {
        match (opt_code, opt_lang) {
            (Some(code), Some(lang)) => {
                let count = vendored_loc::comment_frac(code, lang);
                comment_lines_fracs.push(Some(
                    count.comment as f32 / count.lines.max(1) as f32,
                ));
                let total_chars = count.comment_chars + count.code_chars;
                comment_chars_fracs.push(Some(
                    count.comment_chars as f32 / total_chars.max(1) as f32,
                ));
            }
            _ => {
                comment_lines_fracs.push(None);
                comment_chars_fracs.push(None);
            }
        }
    }

    df.with_column(Series::new(
        "ors_comment_lines_frac".into(),
        ChunkedArray::<Float32Type>::from_iter(comment_lines_fracs),
    )).unwrap();

    df.with_column(Series::new(
        "ors_comment_chars_frac".into(),
        ChunkedArray::<Float32Type>::from_iter(comment_chars_fracs),
    )).unwrap();

    Ok(())
}

#[register]
fn decontaminate(args: DecontaminateArgs, df: &mut DataFrame) -> PyResult<()> {
    for (cur_label, cur_ngrams) in args.ngrams.into_iter() {
        let normalized_ngrams = cur_ngrams
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<String>>();
        let ngrams: HashSet<Vec<&str>> = HashSet::from_iter(
            normalized_ngrams.iter().map(|s| s.split(' ').collect::<Vec<&str>>()),
        );

        let colname = format!("{cur_label}_matched_ngrams");
        df.with_column::<ChunkedArray<UInt64Type>>(
            get_code_col(df)
                .to_lowercase()
                .replace_all(r"\s+", " ")
                .unwrap()
                .apply_nonnull_values_generic(DataType::UInt64, |s| {
                    s.split(' ')
                        .collect::<Vec<&str>>()
                        .windows(args.ngram_order)
                        .filter(|gram| ngrams.contains(&gram.to_vec()))
                        .count() as u64
                })
                .with_name(colname.into()),
        ).unwrap();
    }
    Ok(())
}

#[register]
fn ngrams_matches(args: DecontaminateArgs, df: &mut DataFrame) -> PyResult<()> {
    for (cur_label, cur_ngrams) in args.ngrams.into_iter() {
        let normalized_ngrams = cur_ngrams
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<String>>();
        let ngrams: HashSet<Vec<&str>> = HashSet::from_iter(
            normalized_ngrams.iter().map(|s| s.split(' ').collect::<Vec<&str>>()),
        );

        let colname = format!("{cur_label}_ngrams_matches");
        df.with_column(Series::new(
            colname.into(),
            get_code_col(df)
                .to_lowercase()
                .replace_all(r"\s+", " ")
                .unwrap()
                .iter()
                .map(|opt_text| {
                    opt_text.map(|s| {
                        s.split(' ')
                            .collect::<Vec<&str>>()
                            .windows(args.ngram_order)
                            .filter_map(|gram| {
                                let joined = gram.join(" ");
                                if ngrams.contains(&gram.to_vec()) {
                                    Some(joined)
                                } else {
                                    None
                                }
                            })
                            .collect::<Series>()
                    })
                })
                .collect::<ChunkedArray<ListType>>(),
        )).unwrap();
    }
    Ok(())
}
