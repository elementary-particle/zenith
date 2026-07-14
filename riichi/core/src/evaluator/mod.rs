//! Internal four-player hand evaluator.
//!
//! This module is derived from the Apache-2.0 RiichiEnv evaluator pinned by
//! the conformance profile. It is kept private so Zenith's public API and
//! rules model remain independently versioned.

#![allow(dead_code)]

pub(crate) mod agari;
pub(crate) mod hand_evaluator;
pub(crate) mod score;
pub(crate) mod types;
pub(crate) mod yaku;
