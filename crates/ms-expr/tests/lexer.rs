use ms_expr::{lex as lex_raw, CompileError, Tok, Token};

fn lex(s: &str) -> Vec<Token> {
    lex_raw(s).unwrap()
}
fn lex_err(s: &str) -> CompileError {
    lex_raw(s).unwrap_err()
}
fn nums(s: &str) -> Vec<f64> {
    lex(s)
        .into_iter()
        .filter_map(|t| if let Tok::Num(n) = t.tok { Some(n) } else { None })
        .collect()
}
fn ops(s: &str) -> usize {
    lex(s).iter().filter(|t| matches!(t.tok, Tok::Op(_))).count()
}

#[test] fn numbers() { assert_eq!(nums("1 .5 2.5e3 0x1F $x10"), vec![1.0, 0.5, 2500.0, 31.0, 16.0]); }
#[test] fn constants() { assert_eq!(nums("$pi $e $phi"), vec![std::f64::consts::PI, std::f64::consts::E, 1.618033988749895]); }
#[test] fn idents_lowercased() { assert!(matches!(&lex("Zoom")[0].tok, Tok::Ident(s) if s == "zoom")); }
#[test] fn ops_longest_match() { assert_eq!(ops("+= -= *= /= %= == != <= >= && || = < > + - * / % ^ & | !"), 23); }
#[test] fn comments_skipped() { assert_eq!(lex("a // x\n/* y\n */ b").len(), 3); }
#[test] fn bom_and_crlf() { let t = lex("\u{feff}a;\r\nb"); assert_eq!((t[2].line, t[2].col), (2, 1)); }
#[test] fn bad_char_located() { let e = lex_err("a = 1 @"); assert_eq!((e.line, e.col), (1, 7)); }
