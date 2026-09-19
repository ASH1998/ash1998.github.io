---
layout: post
title: "Rune Goblin: My Build Small Hackathon Winner"
date: 2026-06-14 12:00:00 +0530
permalink: /blog/rune-goblin/
desc: "My winning Hugging Face Build Small project: an AI dungeon crawler that turns hand-drawn runes into spells, with a fine-tuned vision model and a deterministic game engine."
categories: [Hackathons, Machine Learning]
tags: [Rune Goblin, Hugging Face, Build Small, OpenBMB, MiniCPM, Fine-tuning, Gradio, Modal, Codex]
image: /static/portfolio/blog-covers/rune-goblin.png
image_alt: "Rune Goblin RPG title artwork"
---

Rune Goblin, my winning project from the Hugging Face Build Small hackathon, started with a question: what would spell casting feel like if you had to draw the spell yourself?

I wanted the player to learn a small magical language, experiment with combinations, and sometimes get a result they did not quite intend. A confident sketch could become a useful spell. A messy one could still do something, but with a weaker effect, a curse, or an unexpected consequence.

That idea became an AI dungeon crawler built around hand-drawn runes, a fine-tuned vision model, and a game engine that checks every proposed action. Getting those pieces to cooperate was the interesting engineering problem.

## A dungeon that responds to your drawings

Players draw rune glyphs on a canvas. The game interprets the drawing, works out the spell intent, checks it against a spell system called RuneLang, and applies the result to the world. The same interaction can affect a battle, an NPC encounter, a locked door, or a hidden dungeon event.

The campaign includes more than nine maps, fourteen-plus spells, rune customizations, hidden quests, boss encounters, and hero evolution. Exploring the side quests and figuring out the boss mechanics can take roughly four hours.

Boss encounters give the drawing system a purpose beyond visual novelty. Some require a particular rune; others depend on combinations, status effects, or discoveries made elsewhere in the dungeon. There are also shrines, portals, chests, NPC hints, and class-specific advantages to discover.

Ambiguity is part of the mechanic. The player is learning how the world responds to their marks, including what happens when those marks are unclear. That makes recognition quality a game-design decision as well as a model-quality problem.

## Two AI paths, one source of game state

Rune Goblin uses OpenBMB's MiniCPM-V-4.6 model family in two roles. The base model helps with NPC dialogue, dungeon narration, and story beats. A version fine-tuned on the custom RuneLang visual dataset interprets drawn spells and produces structured JSON.

Both paths feed into a deterministic game engine. The model can suggest what a drawing means or how an NPC speaks, but persistent changes to health, inventory, quests, and combat go through the game's rules.

That boundary matters during ordinary interactions too. An NPC can offer a clue without changing quest progress on its own. A shopkeeper can haggle within predefined price bands, while the engine checks the item, final price, and inventory transaction before completing the sale.

The resulting split gives the model room to make the dungeon expressive while keeping a save file consistent across turns.

## What happens when you cast a spell

Each spell turn starts with context. The game packages the drawn or selected runes with a compact snapshot of the situation: player health, the enemy, the room, known weaknesses, inventory hints, and recent story state.

The rune interpreter returns a proposal containing information such as detected runes, confidence, ambiguity, spell name, target, effect, status changes, presentation text, and visual tags. The engine then takes that proposal through several checks:

1. **Parse the response.** Read the JSON and repair malformed output where possible.
2. **Constrain the fields.** Clamp the result to the spell schema so downstream code receives an expected structure.
3. **Resolve RuneLang.** Apply the language's grammar, rune combinations, and ambiguity rules.
4. **Check the encounter.** Account for enemy weaknesses, resistances, and the relevant world rules.
5. **Select allowed actions.** Translate the resolved spell into the small set of actions the engine supports.
6. **Update and render.** Apply the state changes, then show spell text, particles, visual effects, sound cues, and dungeon logs.

Those actions include damage, healing, status effects, unlocking a door, looting a chest, charging a boss pylon, recording a discovery, setting a story flag, or moving between rooms. A model response has to pass through this vocabulary of supported actions to affect the world.

This is the agentic loop at the heart of the game: observe the situation, propose an interpretation, validate it, and act. The validation step connects a flexible model response to a world with durable rules.

## Teaching a vision model RuneLang

The training task was deliberately narrow. I needed a model that could recognize Rune Goblin's spell language: shapes, ordering, combinations, and ambiguity, then connect those observations to valid spell intent.

The offline pipeline began with the RuneLang rulebook. It defined the vocabulary, grammar, combinations, risks, and chaos rules. From there, the pipeline generated synthetic game states and rune combinations, then rendered glyph images with variation in strokes, rotation, scale, noise, and color.

That variation matters because a player drawing on a canvas will not reproduce a clean reference glyph every time. Training examples need to represent imperfect input as well as recognizable spells.

The train and validation records paired images and prompts with target rune metadata, spell presentations, and structured spell JSON. Dataset construction included balanced coverage and ambiguous examples, so the task covered uncertainty as well as straightforward recognition.

The workflow then used LoRA/QLoRA fine-tuning and published the resulting artifacts. The [Rune Goblin visual dataset](https://huggingface.co/datasets/ASHu2/rune_goblin_visual_dataset) and [goblinV1 model artifacts](https://huggingface.co/ASHu2/goblinV1) are available on Hugging Face.

Evaluation covered JSON validity, rune detection accuracy, ambiguity handling, and basic gameplay sanity. These checks address different failure modes: a response might have valid syntax but identify the wrong rune, or recognize the drawing yet propose something the game cannot safely apply. Runtime validation remains necessary even with a fine-tuned model.

## Where Codex helped

Codex helped throughout the work around the model: designing the RuneLang data format, generating synthetic spell examples, building the glyph renderer, adding noisy and ambiguous variations, and keeping the training targets aligned with the runtime schema.

It also helped write evaluation code and work through the fine-tuning workflow, including Modal notebook steps, LoRA/QLoRA configuration, export notes, local inference paths, and GGUF conversion documentation.

Keeping those pieces aligned was a substantial part of the project. The dataset, model output, validator, and game integration all needed to agree on what a spell meant. Changes also had to make their way into the README, deployment scripts, and instructions for running the project again.

One useful distinction in that documentation was between training artifacts and serving artifacts. The training workflow uses safetensors, while the deployed inference path uses a quantized GGUF build. Keeping those roles clear made the path from fine-tuning to a playable game easier to follow.

## Deploying an interactive game without an always-on GPU

The application uses Gradio and FastAPI and can run on a CPU-only Hugging Face Space. Vision inference runs separately on Modal. When a player draws a spell, the application calls the GPU endpoint, receives structured spell JSON, and continues the game through its own engine.

The deployed model uses a Q8_0 GGUF build with a separate multimodal projector on a Modal A10G GPU. The serving configuration includes:

- A CUDA build of `llama-cpp-python` exposing an OpenAI-compatible `/v1/chat/completions` endpoint.
- A Modal Volume named `goblin-gguf-cache` for the model cache.
- Scale-to-zero with `min_containers=0` and a five-minute idle window.
- Up to ten in-flight requests per replica.
- CPU memory and GPU VRAM snapshots to reduce cold-start time.

Cold starts were especially important for this project. During early experiments, loading the full safetensors model with PyTorch took around five minutes. The GGUF serving pipeline with Modal snapshotting brought that down to roughly ten seconds. These are approximate observations from the project, and they refer to startup time rather than the latency of every spell request.

The working MiniCPM-V-4.6 multimodal path used `llama-cpp-python`'s `Llama` and `MTMDChatHandler`. The deployment followed that same path used by the local backend, rather than relying on the standalone `llama-server` binary.

The Modal image builds the runtime with CUDA support, starts the server, sends a warmup multimodal request, and snapshots the loaded state. Later cold starts can restore the model with its weights already in VRAM. Scaling to zero lets the game avoid keeping a private GPU server running when nobody is playing.

## What I took away from the build

The most satisfying part of Rune Goblin is the connection between a player's drawing and a consequence in the dungeon. A sketch can change a fight, reveal a path, or produce a cursed outcome. The model becomes part of the interaction itself.

Making that work required attention to the entire loop: the spell language, training examples, structured outputs, validation, persistent state, and inference startup. Each part affects whether drawing a rune feels like playing a game.

Winning the Build Small hackathon was a great milestone for the project. The engineering lesson I take from it is the value of giving a model a clearly defined role and building the surrounding system carefully enough that its output becomes something a player can rely on, experiment with, and occasionally be surprised by.
