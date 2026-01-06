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
use hyperpolyglot;
use lazy_regex::Regex;
#[cfg(feature = "software_metrics")]
use software_metrics::{get_metrics, has_parsing_errors};
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::io::BufReader;
use tiktoken_rs;

mod vendored_loc;

const TIKTOKEN_O200K_BASE_NAME: &str = "tiktoken_o200k_base";
const GITHUB_O200K_BASE_NAME: &str = "github_o200k_base";

// ============================================================================
// Result Structs
// ============================================================================

/// Basic statistics for a code string
pub struct BasicStatsResult {
    pub num_bytes: u32,
    pub valid_utf8: bool,
    pub max_line_length: Option<u64>,
    pub num_lines: Option<u64>,
    pub average_line_length: Option<f32>,
    pub contains_xml_header: Option<bool>,
    pub alpha_percent: f32,
    pub alnum_percent: f32,
    pub base64_percent: f32,
    pub hex_percent: f32,
    pub unicode_percent: f32,
    pub base64_match_lengths: Vec<u64>,
    pub hex_match_lengths: Vec<u64>,
    pub unicode_match_lengths: Vec<u64>,
}

/// Language detection result
pub struct LanguageResult {
    pub language: Option<String>,
    pub detector: Option<String>,
}

/// Software metrics result
#[cfg(feature = "software_metrics")]
pub struct SoftwareMetricsResult {
    pub cyclomatic_complexity: f32,
    pub cognitive_complexity: f32,
    pub exits_average: f32,
    pub maintainability_index: f32,
    pub halstead_difficulty: f32,
    pub comment_lines: f32,
    pub comment_lines_frac: f32,
    pub comment_lines_per_space: f32,
    pub blank_lines: f32,
    pub blank_lines_per_space: f32,
    pub args_average: f32,
    pub functions_closures_per_space: f32,
    pub total_cda: f32,
    pub total_wmc: f32,
    pub total_coa: f32,
    pub parsed_ok: bool,
}

/// Tokenization result
pub struct TokenizeResult {
    pub tokens: Vec<u32>,
    pub num_tokens: u64,
}

/// OpenCoder-RS statistics result
pub struct OpenCoderRsResult {
    pub comment_lines_frac: Option<f32>,
    pub comment_chars_frac: Option<f32>,
}

// ============================================================================
// Pattern Extraction Helpers
// ============================================================================

fn extract_pattern_stats(code: &str) -> (f32, f32, f32, f32, f32, Vec<u64>, Vec<u64>, Vec<u64>) {
    let base64_pattern = Regex::new(r"[a-zA-Z0-9+/\n=]{64,}").unwrap();
    let hex_pattern = Regex::new(r"(?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}").unwrap();
    let unicode_pattern = Regex::new(r"(?:\\u[0-9a-fA-F]{4}){8,}").unwrap();

    let slen = code.len() as f32;
    if slen == 0.0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, vec![], vec![], vec![]);
    }

    let alpha_percent = (code.chars().filter(|c| c.is_alphabetic()).count() as f32) / slen;
    let alnum_percent = code.chars().filter(|c| c.is_alphanumeric()).count() as f32 / slen;
    let base64_percent =
        base64_pattern.find_iter(code).map(|m| m.len()).sum::<usize>() as f32 / slen;
    let hex_percent = hex_pattern.find_iter(code).map(|m| m.len()).sum::<usize>() as f32 / slen;
    let unicode_percent =
        unicode_pattern.find_iter(code).map(|m| m.len()).sum::<usize>() as f32 / slen;

    let base64_match_lengths: Vec<u64> =
        base64_pattern.find_iter(code).map(|m| m.len() as u64).collect();
    let hex_match_lengths: Vec<u64> = hex_pattern.find_iter(code).map(|m| m.len() as u64).collect();
    let unicode_match_lengths: Vec<u64> = unicode_pattern
        .find_iter(code)
        .map(|m| m.len() as u64)
        .collect();

    (
        alpha_percent,
        alnum_percent,
        base64_percent,
        hex_percent,
        unicode_percent,
        base64_match_lengths,
        hex_match_lengths,
        unicode_match_lengths,
    )
}

// ============================================================================
// Public API Functions
// ============================================================================

/// Compute basic statistics for a list of code strings.
pub fn basic_stats(
    codes: &[String],
    xml_header_search_length: usize,
    max_byte_size: Option<usize>,
) -> Vec<BasicStatsResult> {
    let byte_size_threshold = max_byte_size.unwrap_or(usize::MAX);

    codes
        .iter()
        .map(|code| {
            let num_bytes = code.len() as u32;
            let valid_utf8 = true; // Already a String, so valid UTF-8

            // Check if code exceeds size threshold
            let filtered = code.len() > byte_size_threshold;

            let (
                alpha_percent,
                alnum_percent,
                base64_percent,
                hex_percent,
                unicode_percent,
                base64_match_lengths,
                hex_match_lengths,
                unicode_match_lengths,
            ) = extract_pattern_stats(code);

            if filtered {
                BasicStatsResult {
                    num_bytes,
                    valid_utf8,
                    max_line_length: None,
                    num_lines: None,
                    average_line_length: None,
                    contains_xml_header: None,
                    alpha_percent,
                    alnum_percent,
                    base64_percent,
                    hex_percent,
                    unicode_percent,
                    base64_match_lengths,
                    hex_match_lengths,
                    unicode_match_lengths,
                }
            } else {
                let max_line_length = code.lines().map(|line| line.len()).max().unwrap_or(0) as u64;
                let num_lines = code.lines().count() as u64;
                let average_line_length = if num_lines > 0 {
                    code.len() as f32 / num_lines as f32
                } else {
                    0.0
                };

                let mut search_end = min(xml_header_search_length, code.len());
                while !code.is_char_boundary(search_end) && search_end < code.len() {
                    search_end += 1;
                }
                let contains_xml_header = code[0..search_end].contains("<?xml version=");

                BasicStatsResult {
                    num_bytes,
                    valid_utf8,
                    max_line_length: Some(max_line_length),
                    num_lines: Some(num_lines),
                    average_line_length: Some(average_line_length),
                    contains_xml_header: Some(contains_xml_header),
                    alpha_percent,
                    alnum_percent,
                    base64_percent,
                    hex_percent,
                    unicode_percent,
                    base64_match_lengths,
                    hex_match_lengths,
                    unicode_match_lengths,
                }
            }
        })
        .collect()
}

/// Detect language using hyperpolyglot.
fn detect_single_language(filename: Option<&str>, content: &str) -> (Option<String>, Option<String>) {
    let candidate =
        filename.and_then(|filename| hyperpolyglot::detectors::get_language_from_filename(filename));
    if let Some(candidate) = candidate {
        return (
            Some(candidate.to_string()),
            Some("Filename".to_string()),
        );
    };

    let extension = filename.and_then(|filename| hyperpolyglot::detectors::get_extension(filename));
    let candidates = extension
        .map(|ext| hyperpolyglot::detectors::get_languages_from_extension(ext))
        .unwrap_or_else(Vec::new);

    if candidates.len() == 1 {
        return (
            Some(candidates[0].to_string()),
            Some("Extension".to_string()),
        );
    };

    let mut reader = BufReader::new(content.as_bytes());
    let candidates = match hyperpolyglot::detectors::get_languages_from_shebang(&mut reader) {
        Ok(shebang_langs) => hyperpolyglot::filter_candidates(candidates, shebang_langs),
        Err(_) => candidates,
    };
    if candidates.len() == 1 {
        return (
            Some(candidates[0].to_string()),
            Some("Shebang".to_string()),
        );
    };

    let content_truncated = hyperpolyglot::truncate(content);
    let candidates = if candidates.len() > 1 {
        if let Some(extension) = extension {
            let languages = hyperpolyglot::detectors::get_languages_from_heuristics(
                &extension[..],
                &candidates,
                &content_truncated,
            );
            hyperpolyglot::filter_candidates(candidates, languages)
        } else {
            candidates
        }
    } else {
        candidates
    };

    match candidates.len() {
        1 => (
            Some(candidates[0].to_string()),
            Some("Heuristics".to_string()),
        ),
        _ => {
            let classified = hyperpolyglot::detectors::classify(&content_truncated, &candidates);
            (
                Some(classified.to_string()),
                Some("Classifier".to_string()),
            )
        }
    }
}

/// Detect programming languages for a list of code strings.
pub fn detect_language(codes: &[String], filenames: &[String]) -> Vec<LanguageResult> {
    codes
        .iter()
        .zip(filenames.iter())
        .map(|(code, filename)| {
            let fname = if filename.is_empty() {
                None
            } else {
                Some(filename.as_str())
            };
            let (language, detector) = detect_single_language(fname, code);
            LanguageResult { language, detector }
        })
        .collect()
}

/// Compute software metrics for a list of code strings.
#[cfg(feature = "software_metrics")]
pub fn software_metrics(
    codes: &[String],
    languages: &[Option<String>],
) -> Vec<SoftwareMetricsResult> {
    codes
        .iter()
        .zip(languages.iter())
        .map(|(code, lang_opt)| {
            let mut result = SoftwareMetricsResult {
                cyclomatic_complexity: 0.0,
                cognitive_complexity: 0.0,
                exits_average: 0.0,
                maintainability_index: 0.0,
                halstead_difficulty: 0.0,
                comment_lines: 0.0,
                comment_lines_frac: 0.0,
                comment_lines_per_space: 0.0,
                blank_lines: 0.0,
                blank_lines_per_space: 0.0,
                args_average: 0.0,
                functions_closures_per_space: 0.0,
                total_cda: 0.0,
                total_wmc: 0.0,
                total_coa: 0.0,
                parsed_ok: false,
            };

            if let Some(lang) = lang_opt {
                result.parsed_ok = !has_parsing_errors(code, lang);
                if let Some(mzmetrics) = get_metrics(lang, code) {
                    let metrics = mzmetrics.metrics;
                    result.cyclomatic_complexity = metrics.cyclomatic.cyclomatic_average() as f32;
                    result.cognitive_complexity = metrics.cognitive.cognitive_average() as f32;
                    result.exits_average = metrics.nexits.exit_average() as f32;
                    result.maintainability_index = metrics.mi.mi_visual_studio() as f32;
                    result.halstead_difficulty = metrics.halstead.difficulty() as f32;
                    result.comment_lines = metrics.loc.cloc() as f32;
                    result.comment_lines_frac =
                        (metrics.loc.cloc() / (1. + metrics.loc.sloc())) as f32;
                    result.comment_lines_per_space = metrics.loc.cloc_average() as f32;
                    result.blank_lines = metrics.loc.blank() as f32;
                    result.blank_lines_per_space = metrics.loc.blank_average() as f32;
                    result.args_average = metrics.nargs.nargs_average() as f32;
                    result.functions_closures_per_space = metrics.nom.average() as f32;

                    let cda = metrics.npa.total_cda();
                    if !cda.is_nan() {
                        result.total_cda = cda as f32;
                        result.total_wmc = metrics.wmc.total_wmc() as f32;
                    }
                    let coa = metrics.npm.total_coa();
                    if !coa.is_nan() {
                        result.total_coa = coa as f32;
                    }
                }
            }
            result
        })
        .collect()
}

/// Create a bpe_openai tokenizer from a custom vocabulary.
fn get_tokenizer_from_vocab(
    vocab: &str,
    patterns: Option<&[(String, bool)]>,
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
            ]
            .join("|");
            let pat2 = "\\s+\\s";
            let pat3 = "\\s+";
            bpe_openai::Tokenizer::new_lookahead(bpe, &[(&pat1, false), (pat2, true), (pat3, false)])
                .unwrap()
        }
        Some(pats) => bpe_openai::Tokenizer::new_lookahead(
            bpe,
            &pats
                .iter()
                .map(|(pat, is_lookahead)| (pat.as_str(), *is_lookahead))
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    }
}

/// Tokenize a list of code strings.
pub fn tokenize(
    codes: &[String],
    tokenizer_name: &str,
    vocab: Option<&str>,
    pretokenizer_patterns: Option<&[(String, bool)]>,
) -> Vec<TokenizeResult> {
    if tokenizer_name == TIKTOKEN_O200K_BASE_NAME {
        let tok = tiktoken_rs::o200k_base().unwrap();
        codes
            .iter()
            .map(|code| {
                let tokens: Vec<u32> = tok
                    .encode_with_special_tokens(code)
                    .into_iter()
                    .map(|t| t as u32)
                    .collect();
                let num_tokens = tokens.len() as u64;
                TokenizeResult { tokens, num_tokens }
            })
            .collect()
    } else if tokenizer_name == GITHUB_O200K_BASE_NAME {
        let tokenizer = bpe_openai::o200k_base();
        codes
            .iter()
            .map(|code| {
                let tokens: Vec<u32> = tokenizer.encode(code);
                let num_tokens = tokens.len() as u64;
                TokenizeResult { tokens, num_tokens }
            })
            .collect()
    } else if let Some(v) = vocab {
        let tokenizer = get_tokenizer_from_vocab(v, pretokenizer_patterns);
        codes
            .iter()
            .map(|code| {
                let tokens: Vec<u32> = tokenizer.encode(code);
                let num_tokens = tokens.len() as u64;
                TokenizeResult { tokens, num_tokens }
            })
            .collect()
    } else {
        panic!(
            "Provide a vocabulary for custom tokenizer {}",
            tokenizer_name
        );
    }
}

/// Compute OpenCoder-RS comment statistics.
pub fn opencoder_rs_stats(codes: &[String], languages: &[Option<String>]) -> Vec<OpenCoderRsResult> {
    codes
        .iter()
        .zip(languages.iter())
        .map(|(code, lang_opt)| match lang_opt {
            Some(lang) => {
                let count = vendored_loc::comment_frac(code, lang);
                let comment_lines_frac = count.comment as f32 / count.lines.max(1) as f32;
                let total_chars = count.comment_chars + count.code_chars;
                let comment_chars_frac = count.comment_chars as f32 / total_chars.max(1) as f32;
                OpenCoderRsResult {
                    comment_lines_frac: Some(comment_lines_frac),
                    comment_chars_frac: Some(comment_chars_frac),
                }
            }
            None => OpenCoderRsResult {
                comment_lines_frac: None,
                comment_chars_frac: None,
            },
        })
        .collect()
}

/// Check for n-gram contamination in code strings.
pub fn decontaminate(
    codes: &[String],
    ngrams_map: &HashMap<String, Vec<String>>,
    ngram_order: usize,
) -> HashMap<String, Vec<u64>> {
    let mut results: HashMap<String, Vec<u64>> = HashMap::new();

    for (label, ngram_list) in ngrams_map.iter() {
        let normalized_ngrams: Vec<String> =
            ngram_list.iter().map(|s| s.to_lowercase()).collect();
        let ngrams: HashSet<Vec<&str>> = HashSet::from_iter(
            normalized_ngrams
                .iter()
                .map(|s| s.split(' ').collect::<Vec<&str>>()),
        );

        let counts: Vec<u64> = codes
            .iter()
            .map(|code| {
                let normalized = code.to_lowercase();
                let whitespace_re = lazy_regex::regex!(r"\s+");
                let normalized = whitespace_re.replace_all(&normalized, " ");
                normalized
                    .split(' ')
                    .collect::<Vec<&str>>()
                    .windows(ngram_order)
                    .filter(|gram| ngrams.contains(&gram.to_vec()))
                    .count() as u64
            })
            .collect();

        results.insert(label.clone(), counts);
    }

    results
}

/// Find matching n-grams in code strings.
pub fn ngrams_matches(
    codes: &[String],
    ngrams_map: &HashMap<String, Vec<String>>,
    ngram_order: usize,
) -> HashMap<String, Vec<Vec<String>>> {
    let mut results: HashMap<String, Vec<Vec<String>>> = HashMap::new();

    for (label, ngram_list) in ngrams_map.iter() {
        let normalized_ngrams: Vec<String> =
            ngram_list.iter().map(|s| s.to_lowercase()).collect();
        let ngrams: HashSet<Vec<&str>> = HashSet::from_iter(
            normalized_ngrams
                .iter()
                .map(|s| s.split(' ').collect::<Vec<&str>>()),
        );

        let matches: Vec<Vec<String>> = codes
            .iter()
            .map(|code| {
                let normalized = code.to_lowercase();
                let whitespace_re = lazy_regex::regex!(r"\s+");
                let normalized = whitespace_re.replace_all(&normalized, " ");
                normalized
                    .split(' ')
                    .collect::<Vec<&str>>()
                    .windows(ngram_order)
                    .filter_map(|gram| {
                        if ngrams.contains(&gram.to_vec()) {
                            Some(gram.join(" "))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        results.insert(label.clone(), matches);
    }

    results
}
