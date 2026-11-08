//! Hand-written lexer. Line/column are 1-based and count chars.

use crate::CompileError;

/// A lexical token kind.
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Num(f64),
    /// Identifier, lowercased (the language is case-insensitive).
    Ident(String),
    /// Operator text, e.g. `"+="`.
    Op(&'static str),
    LParen,
    RParen,
    Comma,
    Semi,
    Eof,
}

/// A token with the position of its first char.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
    pub col: u32,
}

/// Two-char operators; tried before the one-char ones (longest match).
const OPS2: [&str; 11] = ["+=", "-=", "*=", "/=", "%=", "==", "!=", "<=", ">=", "&&", "||"];
const OPS1: [&str; 12] = ["=", "<", ">", "+", "-", "*", "/", "%", "^", "&", "|", "!"];

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: u32,
    col: u32,
}

impl Lexer {
    fn peek(&self, off: usize) -> Option<char> {
        self.chars.get(self.pos + off).copied()
    }

    /// Consume one char, tracking position. `\r\n` counts as a single newline.
    fn bump(&mut self) -> Option<char> {
        let c = self.peek(0)?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else if c == '\r' && self.peek(0) == Some('\n') {
            // The following '\n' performs the line advance.
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn err(&self, line: u32, col: u32, message: impl Into<String>) -> CompileError {
        CompileError { line, col, message: message.into() }
    }

    fn skip_trivia(&mut self) -> Result<(), CompileError> {
        loop {
            match (self.peek(0), self.peek(1)) {
                (Some(c), _) if c.is_whitespace() => {
                    self.bump();
                }
                (Some('/'), Some('/')) => {
                    while let Some(c) = self.peek(0) {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                (Some('/'), Some('*')) => {
                    let (l, c) = (self.line, self.col);
                    self.bump();
                    self.bump();
                    loop {
                        match (self.peek(0), self.peek(1)) {
                            (Some('*'), Some('/')) => {
                                self.bump();
                                self.bump();
                                break;
                            }
                            (Some(_), _) => {
                                self.bump();
                            }
                            (None, _) => return Err(self.err(l, c, "unterminated block comment")),
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn take_while(&mut self, f: impl Fn(char) -> bool) -> String {
        let mut s = String::new();
        while let Some(c) = self.peek(0) {
            if !f(c) {
                break;
            }
            s.push(c);
            self.bump();
        }
        s
    }

    fn hex(&self, digits: &str, line: u32, col: u32) -> Result<Tok, CompileError> {
        u64::from_str_radix(digits, 16)
            .map(|n| Tok::Num(n as f64))
            .map_err(|_| self.err(line, col, "invalid hex literal"))
    }

    fn dollar(&mut self, line: u32, col: u32) -> Result<Tok, CompileError> {
        self.bump(); // '$'
        let word = self.take_while(|c| c.is_ascii_alphanumeric() || c == '_').to_lowercase();
        match word.as_str() {
            "pi" => Ok(Tok::Num(std::f64::consts::PI)),
            "e" => Ok(Tok::Num(std::f64::consts::E)),
            "phi" => Ok(Tok::Num(std::f64::consts::GOLDEN_RATIO)),
            _ => match word.strip_prefix('x') {
                Some(d) if !d.is_empty() => self.hex(d, line, col),
                _ => Err(self.err(line, col, format!("unknown constant '${word}'"))),
            },
        }
    }

    fn number(&mut self, line: u32, col: u32) -> Result<Tok, CompileError> {
        if self.peek(0) == Some('0') && matches!(self.peek(1), Some('x' | 'X')) {
            self.bump();
            self.bump();
            let d = self.take_while(|c| c.is_ascii_hexdigit());
            return self.hex(&d, line, col);
        }
        let mut s = self.take_while(|c| c.is_ascii_digit());
        if self.peek(0) == Some('.') {
            self.bump();
            s.push('.');
            s.push_str(&self.take_while(|c| c.is_ascii_digit()));
        }
        if matches!(self.peek(0), Some('e' | 'E')) {
            let sign = matches!(self.peek(1), Some('+' | '-'));
            let digit_at = if sign { 2 } else { 1 };
            if self.peek(digit_at).is_some_and(|c| c.is_ascii_digit()) {
                for _ in 0..digit_at {
                    if let Some(c) = self.bump() {
                        s.push(c);
                    }
                }
                s.push_str(&self.take_while(|c| c.is_ascii_digit()));
            }
        }
        s.parse::<f64>()
            .map(Tok::Num)
            .map_err(|_| self.err(line, col, format!("invalid number '{s}'")))
    }

    fn operator(&self) -> Option<&'static str> {
        let a = self.peek(0)?;
        if let Some(b) = self.peek(1) {
            let pair: String = [a, b].iter().collect();
            if let Some(op) = OPS2.iter().find(|o| **o == pair) {
                return Some(op);
            }
        }
        OPS1.iter().find(|o| o.starts_with(a)).copied()
    }

    fn next_tok(&mut self) -> Result<Tok, CompileError> {
        let (line, col) = (self.line, self.col);
        let Some(c) = self.peek(0) else { return Ok(Tok::Eof) };
        if c == '$' {
            return self.dollar(line, col);
        }
        if c.is_ascii_digit() || (c == '.' && self.peek(1).is_some_and(|d| d.is_ascii_digit())) {
            return self.number(line, col);
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let s = self.take_while(|c| c.is_ascii_alphanumeric() || c == '_');
            return Ok(Tok::Ident(s.to_lowercase()));
        }
        let simple = match c {
            '(' => Some(Tok::LParen),
            ')' => Some(Tok::RParen),
            ',' => Some(Tok::Comma),
            ';' => Some(Tok::Semi),
            _ => None,
        };
        if let Some(t) = simple {
            self.bump();
            return Ok(t);
        }
        if let Some(op) = self.operator() {
            for _ in 0..op.len() {
                self.bump();
            }
            return Ok(Tok::Op(op));
        }
        Err(self.err(line, col, format!("unexpected character '{c}'")))
    }
}

/// Tokenise `src`. The result always ends with a `Tok::Eof` token.
pub fn lex(src: &str) -> Result<Vec<Token>, CompileError> {
    let mut lx = Lexer { chars: src.chars().collect(), pos: 0, line: 1, col: 1 };
    if lx.peek(0) == Some('\u{feff}') {
        lx.pos += 1;
    }
    let mut out = Vec::new();
    loop {
        lx.skip_trivia()?;
        let (line, col) = (lx.line, lx.col);
        let tok = lx.next_tok()?;
        let done = tok == Tok::Eof;
        out.push(Token { tok, line, col });
        if done {
            return Ok(out);
        }
    }
}
