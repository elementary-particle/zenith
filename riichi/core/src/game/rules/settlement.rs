use super::scoring::{round_100, ScoreValue};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Settlement {
    pub deltas: [i32; 4],
    pub dealer_continues: bool,
    pub deposits_after: u16,
}

/// One ordered ron result. Claims must be supplied in discard-relative
/// priority order; the first winner receives the table deposits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RonClaim {
    pub winner: u8,
    pub value: ScoreValue,
    pub liable_seat: Option<u8>,
    /// Base points belonging to the pao-triggering yakuman component only.
    pub liable_base_points: u32,
}

pub fn ron(
    winner: u8,
    loser: u8,
    dealer: u8,
    value: &ScoreValue,
    honba: u16,
    deposits: u16,
) -> Settlement {
    let mut deltas = [0; 4];
    let payment = round_100(value.base_points * if winner == dealer { 6 } else { 4 })
        + i32::from(honba) * 300;
    deltas[winner as usize] += payment + i32::from(deposits) * 1000;
    deltas[loser as usize] -= payment;
    Settlement {
        deltas,
        dealer_continues: winner == dealer,
        deposits_after: 0,
    }
}

pub fn tsumo(winner: u8, dealer: u8, value: &ScoreValue, honba: u16, deposits: u16) -> Settlement {
    let mut deltas = [0; 4];
    apply_tsumo_component(&mut deltas, winner, dealer, value.base_points, honba);
    deltas[winner as usize] += i32::from(deposits) * 1000;
    Settlement {
        deltas,
        dealer_continues: winner == dealer,
        deposits_after: 0,
    }
}

/// Settles every selected ron. Mahjong Soul retains all winners, charges
/// honba for every win, and awards accumulated riichi deposits to the first
/// winner in discard-relative order.
pub fn multiple_ron(
    claims: &[RonClaim],
    loser: u8,
    dealer: u8,
    honba: u16,
    deposits: u16,
) -> Settlement {
    let mut deltas = [0; 4];
    let mut dealer_continues = false;
    for (index, claim) in claims.iter().enumerate() {
        let claim_deposits = if index == 0 { deposits } else { 0 };
        let result = if let Some(liable_seat) = claim.liable_seat {
            pao_ron(
                claim.winner,
                loser,
                dealer,
                &claim.value,
                liable_seat,
                claim.liable_base_points,
                honba,
                claim_deposits,
            )
        } else {
            ron(
                claim.winner,
                loser,
                dealer,
                &claim.value,
                honba,
                claim_deposits,
            )
        };
        for (total, delta) in deltas.iter_mut().zip(result.deltas) {
            *total += delta;
        }
        dealer_continues |= result.dealer_continues;
    }
    Settlement {
        deltas,
        dealer_continues,
        deposits_after: if claims.is_empty() { deposits } else { 0 },
    }
}

/// Applies Mahjong Soul's liability-only pao rule to ron. When a third seat
/// discards the winning tile, that seat and the liable seat split only the
/// pao yakuman component; the liable seat also pays the honba. Other hand
/// components remain the discarder’s responsibility.
#[allow(clippy::too_many_arguments)]
pub fn pao_ron(
    winner: u8,
    loser: u8,
    dealer: u8,
    value: &ScoreValue,
    liable_seat: u8,
    liable_base_points: u32,
    honba: u16,
    deposits: u16,
) -> Settlement {
    debug_assert!(winner < 4 && loser < 4 && dealer < 4 && liable_seat < 4);
    debug_assert_ne!(winner, loser);
    debug_assert_ne!(winner, liable_seat);
    let liable_base = liable_base_points.min(value.base_points);
    if liable_base == 0 {
        return ron(winner, loser, dealer, value, honba, deposits);
    }

    let ordinary = ron_payment(value.base_points - liable_base, winner, dealer);
    let liable = ron_payment(liable_base, winner, dealer);
    let honba_payment = i32::from(honba) * 300;
    let mut deltas = [0; 4];
    if liable_seat == loser {
        deltas[loser as usize] -= ordinary + liable + honba_payment;
    } else {
        let discarder_share = liable / 2;
        deltas[loser as usize] -= ordinary + discarder_share;
        deltas[liable_seat as usize] -= liable - discarder_share + honba_payment;
    }
    deltas[winner as usize] += ordinary + liable + honba_payment + i32::from(deposits) * 1_000;
    Settlement {
        deltas,
        dealer_continues: winner == dealer,
        deposits_after: 0,
    }
}

/// Applies Mahjong Soul's liability-only pao rule to tsumo. The liable seat
/// pays the whole liable yakuman component and honba; non-liable components
/// retain the ordinary split among all opponents.
#[allow(clippy::too_many_arguments)]
pub fn pao_tsumo(
    winner: u8,
    dealer: u8,
    value: &ScoreValue,
    liable_seat: u8,
    liable_base_points: u32,
    honba: u16,
    deposits: u16,
) -> Settlement {
    debug_assert!(winner < 4 && dealer < 4 && liable_seat < 4);
    debug_assert_ne!(winner, liable_seat);
    let liable_base = liable_base_points.min(value.base_points);
    if liable_base == 0 {
        return tsumo(winner, dealer, value, honba, deposits);
    }

    let mut deltas = [0; 4];
    apply_tsumo_component(
        &mut deltas,
        winner,
        dealer,
        value.base_points - liable_base,
        0,
    );
    let liable_payment = ron_payment(liable_base, winner, dealer) + i32::from(honba) * 300;
    deltas[liable_seat as usize] -= liable_payment;
    deltas[winner as usize] += liable_payment + i32::from(deposits) * 1_000;
    Settlement {
        deltas,
        dealer_continues: winner == dealer,
        deposits_after: 0,
    }
}

pub fn exhaustive_draw(tenpai_mask: u8, dealer: u8, deposits: u16) -> Settlement {
    let tenpai = (0..4).filter(|seat| tenpai_mask & (1 << seat) != 0).count() as i32;
    let mut deltas = [0; 4];
    if tenpai > 0 && tenpai < 4 {
        for (seat, delta) in deltas.iter_mut().enumerate() {
            *delta = if tenpai_mask & (1 << seat) != 0 {
                3000 / tenpai
            } else {
                -3000 / (4 - tenpai)
            };
        }
    }
    Settlement {
        deltas,
        dealer_continues: tenpai_mask & (1 << dealer) != 0,
        deposits_after: deposits,
    }
}

pub fn ranking(scores: [i32; 4], initial_seats: [u8; 4]) -> [u8; 4] {
    let mut seats = [0_u8, 1, 2, 3];
    seats.sort_by_key(|&seat| {
        (
            std::cmp::Reverse(scores[seat as usize]),
            initial_seats[seat as usize],
        )
    });
    let mut ranks = [0; 4];
    for (rank, seat) in seats.into_iter().enumerate() {
        ranks[seat as usize] = rank as u8 + 1;
    }
    ranks
}

fn ron_payment(base_points: u32, winner: u8, dealer: u8) -> i32 {
    round_100(base_points * if winner == dealer { 6 } else { 4 })
}

fn apply_tsumo_component(
    deltas: &mut [i32; 4],
    winner: u8,
    dealer: u8,
    base_points: u32,
    honba: u16,
) {
    for payer in 0..4_u8 {
        if payer == winner {
            continue;
        }
        let multiplier = if winner == dealer || payer == dealer {
            2
        } else {
            1
        };
        let payment = round_100(base_points * multiplier) + i32::from(honba) * 100;
        deltas[payer as usize] -= payment;
        deltas[winner as usize] += payment;
    }
}
