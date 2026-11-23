# Draft: Sunk City, a 5k-weight world renderer for VR

(Draft for idea c0469046. To post with a 60–90 s headset capture. Numbers from the repo as of
2026-11-23; replace the bracketed parts.)

Real-time world models are getting bigger: 5B parameters for 720p at 40 fps on a desktop GPU.
Sunk City goes the other way. Its renderer is a per-pixel net with 4,920 weights, running
as a plain fragment shader in stereo at 90 Hz on a laptop iGPU, and aimed at the Steam Frame.

How it works:
- **Reproject, then refine.** Each display frame, the previous state (colour, depth, world
  velocity, certainty, 4 learned channels) is splatted into the new head pose for both eyes.
  Head motion is therefore exact and instant; the net never has to learn the warp.
- **The net fills and animates.** Disoccluded holes get a hole-aware push-pull fill, then a
  tiny MLP reads 15 taps from a 5-level pyramid and updates every pixel by a bounded Euler
  step of the elapsed time (it runs at its own rate, e.g. 24 Hz).
- **The only geometry** is a field of skyscraper blocks, blended into the state before the net
  runs. The raw render is never shown; the rest is the image's own memory.
- **Trained on ordinary video**: depth (Depth Anything V2) and flow (RAFT) give the state,
  and the net learns to continue free-running rollouts.

Numbers: net 6.3 ms for 2×512² in fp16 on a Vega iGPU (700 MHz); over 96 free-running frames
the net beats a plain push-pull fill on disoccluded pixels by about 12% [update with the
current best run].

Toys in the demo: a day/night fader (at night the blocks become another world), swinging on
ropes between blocks, and a time-stop on the grip that freezes the net's world while you
keep moving.

Code: [repo link]. [Contact / what you're looking for: GPU time, collaborators, funding.]
