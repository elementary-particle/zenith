# RiichiEnv scoring corpus

`agari_4p.json` is an unmodified copy of
`riichienv-core/benches/data/agari_4p.json` from `smly/RiichiEnv` revision
`b1d08b3615a710f929679fefb50d1c384f2070b9`, Git blob
`a3ed3e0625216630d7e83a7b36b3f98000ff3705`.

RiichiEnv is licensed under the Apache License 2.0. The upstream license is available at
<https://github.com/smly/RiichiEnv/blob/b1d08b3615a710f929679fefb50d1c384f2070b9/LICENSE>.

The vendored file SHA-256 is
`0a20e41b90a09b86c656922fb1d2d410ae4839decb8c91d229a09423825b3fac`.

The native scoring adapter uses an attributed private evaluator derived from
the same Apache-2.0 revision. Zenith-owned public types isolate it from the
simulator contract; there is no external `riichienv-core` Cargo or runtime
dependency.
