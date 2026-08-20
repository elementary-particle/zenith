use crate::game::phase::Wind;

pub const STATE_SCHEMA_VERSION: u32 = 5;
pub const EVENT_SCHEMA_VERSION: u32 = 2;
pub const DECISION_SCHEMA_VERSION: u32 = 2;
pub const HAND_EFFICIENCY_VERSION: u32 = 1;
pub const SNAPSHOT_SCHEMA_VERSION: u32 = 3;
pub const RULES_PROFILE_ID: u32 = 2;
pub const RULES_PROFILE: &str = "riichilab-mjsoul-yonma-v1";
pub const TENHOU_RULES_PROFILE_ID: u32 = 3;
pub const TENHOU_RULES_PROFILE: &str = "tenhou-yonma-v1";
pub const RNG_PROFILE_ID: u32 = 1;
pub const RNG_PROFILE: &str = "pcg32-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RulesProfile {
    pub profile_id: u32,
    pub name: &'static str,
    pub source_revision: &'static str,
    pub game_rule_preset: &'static str,
    pub game_mode: &'static str,
    pub starting_points: i32,
    pub return_points: i32,
    pub maximum_extension_wind: Wind,
    pub tobi_below_zero: bool,
    pub red_fives: [u8; 3],
    pub allows_ron_on_ankan_for_kokushi_musou: bool,
    pub kokushi_musou_13machi_double: bool,
    pub suuankou_tanki_double: bool,
    pub junsei_chuurenpoutou_double: bool,
    pub daisuushii_double: bool,
    pub yakuman_pao_is_liability_only: bool,
    pub sanchaho_is_draw: bool,
    pub kuikae_forbidden: bool,
    /// Tenhou reveals kandora for daiminkan and kakan only after the
    /// replacement-turn action. Ankan indicators remain immediate.
    pub delays_open_and_added_kan_dora: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoreCapabilities {
    pub supported_features: &'static [&'static str],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConformanceInventory {
    pub profile_id: u32,
    pub source_revision: &'static str,
    pub out_of_scope_variants: &'static [&'static str],
    pub documented_deviations: &'static [&'static str],
}

pub static RIICHILAB_MJSOUL: RulesProfile = RulesProfile {
    profile_id: RULES_PROFILE_ID,
    name: RULES_PROFILE,
    source_revision: "b1d08b3615a710f929679fefb50d1c384f2070b9",
    game_rule_preset: "default_mjsoul",
    game_mode: "4p-red-half",
    starting_points: 25_000,
    return_points: 30_000,
    maximum_extension_wind: Wind::West,
    tobi_below_zero: true,
    red_fives: [1, 1, 1],
    allows_ron_on_ankan_for_kokushi_musou: true,
    kokushi_musou_13machi_double: true,
    suuankou_tanki_double: true,
    junsei_chuurenpoutou_double: true,
    daisuushii_double: true,
    yakuman_pao_is_liability_only: true,
    sanchaho_is_draw: false,
    kuikae_forbidden: true,
    delays_open_and_added_kan_dora: false,
};

pub static TENHOU_YONMA: RulesProfile = RulesProfile {
    profile_id: TENHOU_RULES_PROFILE_ID,
    name: TENHOU_RULES_PROFILE,
    source_revision: "tenhou-manual-2026-07-28",
    game_rule_preset: "tenhou_default",
    game_mode: "4p-red",
    starting_points: 25_000,
    return_points: 30_000,
    maximum_extension_wind: Wind::West,
    tobi_below_zero: true,
    red_fives: [1, 1, 1],
    allows_ron_on_ankan_for_kokushi_musou: true,
    kokushi_musou_13machi_double: true,
    suuankou_tanki_double: true,
    junsei_chuurenpoutou_double: true,
    daisuushii_double: true,
    yakuman_pao_is_liability_only: true,
    sanchaho_is_draw: true,
    kuikae_forbidden: true,
    delays_open_and_added_kan_dora: true,
};

pub fn by_id(profile_id: u32) -> Option<&'static RulesProfile> {
    match profile_id {
        RULES_PROFILE_ID => Some(&RIICHILAB_MJSOUL),
        TENHOU_RULES_PROFILE_ID => Some(&TENHOU_YONMA),
        _ => None,
    }
}

pub fn by_name(name: &str) -> Option<&'static RulesProfile> {
    match name {
        RULES_PROFILE => Some(&RIICHILAB_MJSOUL),
        TENHOU_RULES_PROFILE => Some(&TENHOU_YONMA),
        _ => None,
    }
}

pub static CORE_CAPABILITIES: CoreCapabilities = CoreCapabilities {
    supported_features: &[
        "four-player-riichi",
        "east-south-with-west-extension",
        "one-red-five-per-suit",
        "simultaneous-reactions",
        "multiple-and-triple-ron",
        "kuikae-forbidden",
        "kokushi-ankan-rob",
        "mahjong-soul-yakuman-and-pao",
        "tobi-and-agari-yame",
    ],
};

pub static CONFORMANCE_INVENTORY: ConformanceInventory = ConformanceInventory {
    profile_id: RULES_PROFILE_ID,
    source_revision: RIICHILAB_MJSOUL.source_revision,
    out_of_scope_variants: &[
        "three-player-mahjong",
        "network-arrival-order",
        "alternative-red-five-counts",
        "alternative-match-lengths",
    ],
    documented_deviations: &[
        "All eligible reactions are collected in one simultaneous joint frame without network arrival times.",
    ],
};
