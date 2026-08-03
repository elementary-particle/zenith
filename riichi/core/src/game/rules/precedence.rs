use crate::game::action::{ActionCandidate, ActionKind};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReactionResolution {
    Ron(Vec<(u8, ActionCandidate)>),
    Call(u8, ActionCandidate),
    AllPass,
}

impl ReactionResolution {
    pub fn ron_winners(&self) -> &[(u8, ActionCandidate)] {
        match self {
            Self::Ron(winners) => winners,
            Self::Call(..) | Self::AllPass => &[],
        }
    }

    pub fn call(&self) -> Option<(u8, &ActionCandidate)> {
        match self {
            Self::Call(seat, action) => Some((*seat, action)),
            Self::Ron(_) | Self::AllPass => None,
        }
    }
}

pub fn resolve(source: u8, selected: &[(u8, ActionCandidate)]) -> ReactionResolution {
    let mut ron = selected
        .iter()
        .filter(|(_, action)| action.kind == ActionKind::Ron)
        .cloned()
        .collect::<Vec<_>>();
    if !ron.is_empty() {
        ron.sort_by_key(|(seat, _)| (*seat + 4 - source) % 4);
        return ReactionResolution::Ron(ron);
    }

    selected
        .iter()
        .filter(|(_, action)| matches!(action.kind, ActionKind::OpenKan | ActionKind::Pon))
        .min_by_key(|(seat, _)| (*seat + 4 - source) % 4)
        .cloned()
        .or_else(|| {
            selected
                .iter()
                .find(|(_, action)| action.kind == ActionKind::Chi)
                .cloned()
        })
        .map_or(ReactionResolution::AllPass, |(seat, action)| {
            ReactionResolution::Call(seat, action)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::action::ABSENT;

    fn action(kind: ActionKind, source: u8) -> ActionCandidate {
        ActionCandidate {
            kind,
            primary_tile_type: 0,
            source_seat: source,
            tile_count: 0,
            tiles: [ABSENT; 4],
            aux: 0,
            flags: 0,
        }
    }

    #[test]
    fn resolution_is_independent_of_joint_row_order() {
        let selections = vec![
            (0, action(ActionKind::Ron, 1)),
            (2, action(ActionKind::Ron, 1)),
            (3, action(ActionKind::Pon, 1)),
        ];
        let mut reversed = selections.clone();
        reversed.reverse();
        assert_eq!(resolve(1, &selections), resolve(1, &reversed));
    }

    #[test]
    fn nearest_equal_call_wins_and_passes_are_ignored() {
        let result = resolve(
            1,
            &[
                (0, action(ActionKind::Pon, 1)),
                (3, action(ActionKind::Pass, 1)),
                (2, action(ActionKind::OpenKan, 1)),
            ],
        );
        assert_eq!(result.call().map(|(seat, _)| seat), Some(2));
        assert_eq!(
            resolve(
                1,
                &[
                    (0, action(ActionKind::Pass, 1)),
                    (2, action(ActionKind::Pass, 1)),
                    (3, action(ActionKind::Pass, 1)),
                ]
            ),
            ReactionResolution::AllPass
        );
    }
}
