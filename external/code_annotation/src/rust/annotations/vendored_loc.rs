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

//! Custom LOC counting implementation.
//!
//! This is a modified version of the loc crate that works from strings
//! with language specified at runtime.
//! See: https://docs.rs/loc/latest/src/loc/lib.rs.html

use smallvec::{smallvec, SmallVec};
use std::cmp::{max, min};

#[derive(Debug, PartialEq, Default, Clone)]
pub struct Count {
    pub code: u32,
    pub comment: u32,
    pub blank: u32,
    pub lines: u32,
    pub code_chars: usize,
    pub comment_chars: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub enum Lang {
    ActionScript, Ada, Agda, AmbientTalk, Asp, AspNet, Assembly, Autoconf, Awk, Batch,
    BourneShell, C, CCppHeader, CMake, CSharp, CShell, Clojure, CoffeeScript, ColdFusion,
    ColdFusionScript, Coq, Cpp, Css, CUDA, CUDAHeader, D, Dart, DeviceTree, Docker, Elixir,
    Elm, Erlang, Forth, FortranLegacy, FortranModern, FSharp, Gherkin, Glsl, Go, Groovy,
    Handlebars, Haskell, Hex, Html, INI, Idris, IntelHex, Isabelle, Jai, Java, JavaScript,
    Json, Jsx, Julia, Kotlin, Less, LinkerScript, Lean, Lisp, Lua, Make, Makefile, Markdown,
    Mustache, Nim, Nix, OCaml, ObjectiveC, ObjectiveCpp, OpenCl, Oz, Pascal, Perl, Php,
    Polly, PowerShell, Prolog, Protobuf, Puppet, PureScript, Pyret, Python, Qcl, Qml, R,
    Razor, ReStructuredText, Ruby, RubyHtml, Rust, SaltStack, Sass, Scala, Sml, Sql, Stylus,
    Swift, Tcl, Terraform, Tex, Text, Toml, TypeScript, Tsx, UnrealScript, VimScript,
    Wolfram, XML, Yacc, Yaml, Zig, Zsh, Haxe, Unrecognized,
}

use self::Lang::*;

impl Lang {
    pub fn from_name(name: &str) -> Option<Lang> {
        let lang = match name {
            "ActionScript" => ActionScript, "Ada" => Ada, "Agda" => Agda,
            "AmbientTalk" => AmbientTalk, "ASP" => Asp, "ASP.NET" => AspNet,
            "Assembly" => Assembly, "Autoconf" => Autoconf, "Awk" => Awk,
            "Batch" => Batch, "Bourne Shell" => BourneShell, "C" => C,
            "C/C++ Header" => CCppHeader, "CMake" => CMake, "C#" => CSharp,
            "C Shell" => CShell, "Clojure" => Clojure, "CoffeeScript" => CoffeeScript,
            "ColdFusion" => ColdFusion, "ColdFusionScript" => ColdFusionScript,
            "Coq" => Coq, "C++" => Cpp, "CSS" => Css, "CUDA" => CUDA,
            "CUDA Header" => CUDAHeader, "D" => D, "Dart" => Dart,
            "DeviceTree" => DeviceTree, "Docker" => Docker, "Elixir" => Elixir,
            "Elm" => Elm, "Erlang" => Erlang, "Forth" => Forth,
            "FORTRAN Legacy" => FortranLegacy, "FORTRAN Modern" => FortranModern,
            "F#" => FSharp, "Gherkin" => Gherkin, "GLSL" => Glsl, "Go" => Go,
            "Groovy" => Groovy, "Handlebars" => Handlebars, "Haskell" => Haskell,
            "Hex" => Hex, "HTML" => Html, "INI" => INI, "Idris" => Idris,
            "Intel Hex" => IntelHex, "Isabelle" => Isabelle, "Jai" => Jai,
            "Java" => Java, "JavaScript" => JavaScript, "JSON" => Json, "Jsx" => Jsx,
            "Julia" => Julia, "Kotlin" => Kotlin, "Less" => Less,
            "LinkerScript" => LinkerScript, "Lean" => Lean, "Lisp" => Lisp,
            "Lua" => Lua, "Make" => Make, "Makefile" => Makefile, "Markdown" => Markdown,
            "Mustache" => Mustache, "Nim" => Nim, "Nix" => Nix, "OCaml" => OCaml,
            "Objective-C" => ObjectiveC, "Objective-C++" => ObjectiveCpp,
            "OpenCL" => OpenCl, "Oz" => Oz, "Pascal" => Pascal, "Perl" => Perl,
            "PHP" => Php, "Polly" => Polly, "PowerShell" => PowerShell,
            "Prolog" => Prolog, "Protobuf" => Protobuf, "Puppet" => Puppet,
            "PureScript" => PureScript, "Pyret" => Pyret, "Python" => Python,
            "Qcl" => Qcl, "Qml" => Qml, "R" => R, "Razor" => Razor,
            "reStructuredText" => ReStructuredText, "Ruby" => Ruby,
            "RubyHtml" => RubyHtml, "Rust" => Rust, "SaltStack" => SaltStack,
            "Sass" => Sass, "Scala" => Scala, "SML" => Sml, "SQL" => Sql,
            "Stylus" => Stylus, "Swift" => Swift, "Tcl" => Tcl,
            "Terraform" => Terraform, "TeX" => Tex, "Plain Text" => Text,
            "Toml" => Toml, "TypeScript" => TypeScript, "Typescript JSX" => Tsx,
            "UnrealScript" => UnrealScript, "VimL" => VimScript, "Wolfram" => Wolfram,
            "XML" => XML, "Yacc" => Yacc, "YAML" => Yaml, "Zig" => Zig,
            "Z Shell" => Zsh, "Haxe" => Haxe, _ => Unrecognized,
        };
        match lang {
            Unrecognized => None,
            _ => Some(lang),
        }
    }
}

pub fn counter_config_for_lang(lang: Lang) -> (SmallVec<[&'static str; 3]>, SmallVec<[(&'static str, &'static str); 3]>) {
    let c_style = (smallvec!["//"], smallvec![("/*", "*/")]);
    let html_style = (smallvec![], smallvec![("<!--", "-->")]);
    let ml_style = (smallvec![], smallvec![("(*", "*)")]);
    let no_comments = (smallvec![], smallvec![]);
    let prolog_style = (smallvec!["%"], smallvec![("/*", "*/")]);
    let sh_style = (smallvec!["#"], smallvec![]);

    match lang {
        Ada => (smallvec!["--"], smallvec![]),
        Batch => (smallvec!["REM"], smallvec![]),
        Erlang | Tex => (smallvec!["%"], smallvec![]),
        FortranModern => (smallvec!["!"], smallvec![]),
        INI => (smallvec![";"], smallvec![]),
        Protobuf | Zig => (smallvec!["//"], smallvec![]),
        VimScript => (smallvec!["\""], smallvec![]),
        Terraform | Nix => (smallvec!["#"], smallvec![("/*", "*/")]),
        Assembly => (smallvec!["#"], smallvec![("/*", "*/")]),
        CMake => (smallvec!["#"], smallvec![("#[[", "]]")]),
        CoffeeScript => (smallvec!["#"], smallvec![("###", "###")]),
        D => (smallvec!["//"], smallvec![("/*", "*/")]),
        Docker => (smallvec!["#"], smallvec![]),
        Forth => (smallvec!["\\"], smallvec![("(", ")")]),
        FSharp => (smallvec!["//"], smallvec![("(*", "*)")]),
        Julia => (smallvec!["#"], smallvec![("#=", "=#")]),
        Lisp => (smallvec![";"], smallvec![("#|", "|#")]),
        Lean => (smallvec!["--"], smallvec![("/-", "-/")]),
        Lua => (smallvec!["--"], smallvec![("--[[", "]]")]),
        Perl => (smallvec!["#"], smallvec![("=pod", "=cut")]),
        Puppet => (smallvec!["#"], smallvec![]),
        Pyret => (smallvec!["#"], smallvec![("#|", "|#")]),
        Python => (smallvec!["#"], smallvec![("'''", "'''"), ("\"\"\"", "\"\"\"")]),
        Ruby => (smallvec!["#"], smallvec![("=begin", "=end")]),
        Sql => (smallvec!["--"], smallvec![("/*", "*/")]),
        Haskell | Idris | Agda | PureScript | Elm => (smallvec!["--"], smallvec![("{-", "-}")]),
        ColdFusion => (smallvec![], smallvec![("<!---", "--->")]),
        Mustache => (smallvec![], smallvec![("{{!", "}}")]),
        Asp => (smallvec!["'", "REM"], smallvec![]),
        AspNet => (smallvec![], smallvec![("<!--", "-->"), ("<%--", "-->")]),
        Autoconf => (smallvec!["#", "dnl"], smallvec![]),
        Clojure => (smallvec![";", "#"], smallvec![]),
        FortranLegacy => (smallvec!["c", "C", "!", "*"], smallvec![]),
        Handlebars => (smallvec![], smallvec![("<!--", "-->"), ("{{!", "}}")]),
        Php => (smallvec!["#", "//"], smallvec![("/*", "*/")]),
        PowerShell => (smallvec!["#"], smallvec![("<#", "#>")]),
        Isabelle => (smallvec!["--"], smallvec![("{*", "*}"), ("(*", "*)"), ("‹", "›"), ("\\<open>", "\\<close>")]),
        Razor => (smallvec![], smallvec![("<!--", "-->"), ("@*", "*@")]),
        Pascal => (smallvec!["//", "(*"], smallvec![("{", "}")]),
        Text | Markdown | Json | IntelHex | Hex | ReStructuredText => no_comments,
        Oz | Prolog => prolog_style,
        Coq | Sml | Wolfram | OCaml => ml_style,
        Html | Polly | RubyHtml | XML => html_style,
        BourneShell | Make | Awk | CShell | Gherkin | Makefile | Nim | R | SaltStack | Tcl | Toml | Yaml | Zsh | Elixir => sh_style,
        AmbientTalk | C | CCppHeader | Rust | Yacc | ActionScript | ColdFusionScript | Css | Cpp | CUDA | CUDAHeader | CSharp | Dart | DeviceTree | Glsl | Go | Jai | Java | JavaScript | Jsx | Kotlin | Less | LinkerScript | ObjectiveC | ObjectiveCpp | OpenCl | Qcl | Sass | Scala | Swift | TypeScript | Tsx | UnrealScript | Stylus | Qml | Haxe | Groovy => c_style,
        Unrecognized => unreachable!(),
    }
}

pub fn comment_frac(code: &str, lang: &str) -> Count {
    match Lang::from_name(lang) {
        None => Count::default(),
        Some(lang) => {
            let (singles, multis) = counter_config_for_lang(lang);
            let mut c = Count::default();
            let mut multi_stack: Vec<(&str, &str)> = vec![];

            'line: for line in code.lines() {
                c.lines += 1;
                let line = line.trim_start();
                let line_char_len = line.chars().count();

                if line.is_empty() {
                    c.blank += 1;
                    continue;
                }

                if multi_stack.is_empty() {
                    for single_start in singles.iter() {
                        if line.starts_with(single_start) {
                            if multis.iter().any(|(m_start, _)| line.starts_with(m_start)) {
                                break;
                            }
                            c.comment += 1;
                            c.comment_chars += line_char_len;
                            continue 'line;
                        }
                    }

                    if multis.is_empty() {
                        c.code += 1;
                        c.code_chars += line_char_len;
                        continue 'line;
                    }
                }

                if multi_stack.is_empty()
                    && !multis.iter().any(|(start, end)| line.contains(start) || line.contains(end))
                {
                    c.code += 1;
                    c.code_chars += line_char_len;
                    continue 'line;
                }

                let mut pos = 0;
                let mut found_code = 0;
                let line_len = line.len();
                let contains_utf8 = (0..line_len).any(|i| !line.is_char_boundary(i));

                'outer: while pos < line_len {
                    for multi in multis.iter() {
                        let (start, end) = multi;
                        let start_len = start.len();
                        let end_len = end.len();

                        if contains_utf8 {
                            for i in pos..pos + min(max(start_len, end_len) + 1, line_len - pos) {
                                if !line.is_char_boundary(i) {
                                    pos += 1;
                                    continue 'outer;
                                }
                            }
                        }

                        if pos + start_len <= line_len
                            && &line[pos..pos + start_len] == *start
                            && (start != end || multi_stack.is_empty())
                        {
                            pos += start_len;
                            multi_stack.push(*multi);
                            continue;
                        }

                        if !multi_stack.is_empty() {
                            let (_, end) = multi_stack.last().expect("stack last");
                            let end_len = end.len();
                            if pos + end_len <= line_len && &line[pos..pos + end_len] == *end {
                                multi_stack.pop();
                                pos += end_len;
                            }
                        } else if multi_stack.is_empty()
                            && pos < line_len
                            && !line[pos..pos + 1]
                                .chars()
                                .next()
                                .expect("whitespace check")
                                .is_whitespace()
                        {
                            found_code += 1;
                        }
                    }
                    pos += 1;
                }

                if found_code >= multis.len() {
                    c.code += 1;
                    c.code_chars += line_char_len;
                } else {
                    c.comment += 1;
                    c.comment_chars += line_char_len;
                }
            }

            c
        }
    }
}
