use std::fs;

use riichi_core::game::{
    phase::{MeldKind, Wind},
    rules::scoring::{evaluate_hand, WinningContext},
    state::Meld,
};
use serde_json::Value;

fn u8_values(value: &Value) -> Vec<u8> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_u64().expect("integer") as u8)
        .collect()
}

fn wind(value: &Value) -> Wind {
    match value.as_u64().expect("wind") {
        0 => Wind::East,
        1 => Wind::South,
        2 => Wind::West,
        3 => Wind::North,
        value => panic!("invalid wind {value}"),
    }
}

fn meld(value: &Value) -> Meld {
    let values = u8_values(&value["tiles"]);
    let mut tiles = [u8::MAX; 4];
    tiles[..values.len()].copy_from_slice(&values);
    Meld {
        kind: match value["meld_type"].as_str().expect("meld type") {
            "chi" => MeldKind::Chi,
            "pon" => MeldKind::Pon,
            "daiminkan" => MeldKind::OpenKan,
            "ankan" => MeldKind::ClosedKan,
            "kakan" => MeldKind::AddedKan,
            value => panic!("invalid meld type {value}"),
        },
        tiles,
        tile_count: values.len() as u8,
        called_tile: value
            .get("called_tile")
            .and_then(Value::as_u64)
            .map_or(u8::MAX, |value| value as u8),
        from_seat: value
            .get("from_who")
            .and_then(Value::as_u64)
            .map_or(u8::MAX, |value| value as u8),
        created_sequence: 0,
    }
}

#[test]
fn pinned_riichilab_scoring_corpus_matches_exactly() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/riichilab_mjsoul/agari_4p.json"
    );
    let root: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("scoring fixture")).expect("json");
    let cases = root["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 816);

    for (index, case) in cases.iter().enumerate() {
        let conditions = &case["conditions"];
        let context = WinningContext {
            tsumo: conditions["tsumo"].as_bool().expect("tsumo"),
            riichi: conditions["riichi"].as_bool().expect("riichi"),
            double_riichi: conditions["double_riichi"]
                .as_bool()
                .expect("double riichi"),
            ippatsu: conditions["ippatsu"].as_bool().expect("ippatsu"),
            haitei: conditions["haitei"].as_bool().expect("haitei"),
            houtei: conditions["houtei"].as_bool().expect("houtei"),
            rinshan: conditions["rinshan"].as_bool().expect("rinshan"),
            chankan: conditions["chankan"].as_bool().expect("chankan"),
            first_turn_tsumo: conditions["tsumo_first_turn"]
                .as_bool()
                .expect("first turn"),
            seat_wind: wind(&conditions["player_wind"]),
            round_wind: wind(&conditions["round_wind"]),
        };
        let melds = case["melds"]
            .as_array()
            .expect("melds")
            .iter()
            .map(meld)
            .collect::<Vec<_>>();
        let evaluation = evaluate_hand(
            &u8_values(&case["tiles_136"]),
            &melds,
            case["win_tile_136"].as_u64().expect("win tile") as u8,
            &u8_values(&case["dora_indicators"]),
            &u8_values(&case["ura_indicators"]),
            &context,
        );
        let expected = &case["expected"];
        assert_eq!(
            (
                evaluation.is_win,
                evaluation.han,
                evaluation.fu,
                evaluation.yaku_ids
            ),
            (
                expected["is_win"].as_bool().expect("is win"),
                expected["han"].as_u64().expect("han") as u8,
                expected["fu"].as_u64().expect("fu") as u16,
                u8_values(&expected["yaku"])
            ),
            "scoring fixture case {index}"
        );
    }
}
