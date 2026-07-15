"""Shared Veo prompt construction — the pure "given panel + scene/casting
context, build the exact text to send Veo" half of video rendering, with
zero filesystem I/O beyond resolving `image_path`s already known to exist,
and zero render-loop/continuity state (that lives in `pipeline.py`, which
decides WHICH images to attach before calling into this module).

Extracted from `pipeline.py` so the SAME formula construction is available
to any caller that can supply a panel dict + casting/scene context — not
just the main pipeline's storyboard-driven render. `reel/fountain.py`'s
standalone `render` CLI path (screenplay.fountain + cinematography.json,
bypassing storyboard.json) is the second real caller this split was built
for: previously it had its own separate, staler prompt-formula
implementation (pre-dating the five-part-formula rebuild) that was mostly
DEAD CODE for actual rendering anyway, since `pipeline._render_scene_frames`
(shared by both entry points) always rebuilds the prompt from THIS module
regardless of which caller produced the storyboard-shaped input — the real
bug was that `fountain.to_storyboard`'s board didn't populate the fields
this module's formula needs (location, visual_overview, audio_overview,
structured dialogue, camera_angle/movement/lens, multi-character
characters_in_frame), not that the formula itself was duplicated where it
actually mattered. See PROGRESS.md's 2026-07-14 session log for the full
audit trail.

Visual half always follows the fixed five-part formula, in this order, per
Google's official Veo 3.1 prompting guide:
  [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]
Audio half is built via `veo_guide`'s construction helpers (ambient/SFX/
music/dialogue/no-subtitles cues) — that module owns the guide-vocabulary
knowledge; this one only gathers panel/scene data and hands it off.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import veo_guide

# Veo prompting guide — lens/framing terms by shot type. "portrait" enhances
# facial detail on close-ups; "macro lens" suits inserts. Keyed by both
# abbreviations (storyboard schema) and natural language variants
# (cinematography agent / storyboard agent prose output). Depth-of-field
# terms (deep/shallow focus) are intentionally NOT handled here — left to
# whatever the source content already says. Values carry ONLY the framing
# hint, not shot-type wording (e.g. not "extreme close-up") — the shot-type
# label itself (_VEO_SHOT_LABEL below) already supplies that.
VEO_FOCUS: dict[str, str] = {
    # standard abbreviations
    "ECU": "portrait",
    "CU": "portrait",
    "INSERT": "macro lens",
    # natural language (model may output these) — covers both the
    # cinematography agent's spelled-out shot types and the storyboard
    # agent's own abbreviation schema.
    "EXTREME-CLOSE-UP": "portrait",
    "EXTREME CLOSE-UP": "portrait",
    "EXTREME CLOSE UP": "portrait",
    "CLOSE-UP": "portrait",
    "CLOSE UP": "portrait",
}

# Shot-type abbreviation -> natural-language label, for [Cinematography] in
# _panel_cinematography. Values already spelled out in full (e.g. "wide shot")
# pass through unchanged.
VEO_SHOT_LABEL: dict[str, str] = {
    "ECU": "extreme close-up",
    "CU": "close-up",
    "MCU": "medium close-up",
    "MS": "medium shot",
    "FS": "full shot",
    "WS": "wide shot",
    "ELS": "extreme long shot",
    "2S": "two-shot",
    "OTS": "over-the-shoulder shot",
    "POV": "POV shot",
    "INSERT": "insert shot",
}


def panel_cinematography(panel: dict) -> str:
    """[Cinematography] — camera work and shot composition, built fresh from
    the panel's own structured fields (shot_type/camera_angle/camera_movement/
    lens), plus the shot-type-appropriate framing hint (VEO_FOCUS: "portrait"
    for close-ups, "macro lens" for inserts)."""
    shot = (panel.get("shot_type") or "").strip()
    angle = (panel.get("camera_angle") or "").strip()
    movement = (panel.get("camera_movement") or "").strip()
    lens = (panel.get("lens") or "").strip()
    bits: list[str] = []
    if shot:
        label = VEO_SHOT_LABEL.get(shot.upper())
        if not label:
            base_label = shot.capitalize() if shot.isupper() else shot
            label = base_label if base_label.lower().endswith(("shot", "up", "view")) else f"{base_label} shot"
        bits.append(label)
    if angle:
        bits.append(angle.lower())
    if movement:
        bits.append("static camera" if movement.lower() == "static" else movement.lower())
    if lens:
        bits.append(lens if "lens" in lens.lower() else f"{lens} lens")

    hint_terms = VEO_FOCUS.get(shot.upper(), "")
    if hint_terms:
        joined_lower = ", ".join(bits).lower()
        key = hint_terms.split(",")[0].strip()
        if key not in joined_lower:
            bits.append(hint_terms)

    return ", ".join(b for b in bits if b)


def panel_subject(panel: dict, casting_lookup: dict[str, dict],
                  anchored: set[str] | None = None) -> str:
    """[Subject] — every character in frame. A character already grounded by
    an actual seed/reference image THIS call (`anchored` — see
    `anchored_character_names`) gets a short referential mention instead of
    the full `physical_form` text: Veo already sees the real image, so
    restating the description in words is redundant and risks the text and
    the image disagreeing. A character with no resolvable image this call
    still gets the full locked description — for them, the text IS the only
    grounding Veo has (falls back to the bare name if casting has no
    matching entry at all, e.g. an uncast background figure)."""
    names = panel.get("characters_in_frame") or []
    anchored = anchored or set()
    subjects: list[str] = []
    for name in names:
        entry = casting_lookup.get(name)
        if not entry:
            subjects.append(name)
            continue
        if name in anchored:
            subjects.append(f"{name} (as shown in the reference image)")
            continue
        ch = entry.get("character", entry)
        form = (ch.get("physical_form") or "").strip()
        subjects.append(f"{name} ({form})" if form else name)
    return " and ".join(subjects)


def anchored_character_names(names: list[str], casting_lookup: dict[str, dict],
                             out: Path, seed: Path | None,
                             reference_images: list[Path] | None) -> set[str]:
    """Which of THIS panel's in-frame characters already have an actual
    image of themselves attached to this Veo call (`seed` and/or
    `reference_images`) — so `panel_subject` can skip re-describing their
    appearance in words.

    Reverse-maps `seed`/`reference_images` against casting.json's own
    `image_path` by path identity, rather than threading name<->path
    pairing through the caller's own image-resolution logic — both already
    resolve the same portraits for this exact call, so comparing resolved
    paths here reuses that result instead of a second name-resolution pass.

    A `seed` that ISN'T any character's casting portrait is presumed to be
    continuity footage (e.g. the previous clip's last frame, not a fresh
    identity anchor) — continuity is only ever used when the in-frame cast
    hasn't changed from the previous panel, so that image is presumed to
    still show every currently-listed character."""
    if not names:
        return set()
    portrait_paths: dict[str, Path] = {}
    for name in names:
        entry = casting_lookup.get(name)
        if not entry:
            continue
        ch = entry.get("character", entry)
        rel = ch.get("image_path") or entry.get("image_path")
        if rel:
            portrait_paths[name] = (out / rel).resolve()

    anchored: set[str] = set()
    ref_resolved = {p.resolve() for p in (reference_images or [])}
    for name, p in portrait_paths.items():
        if p in ref_resolved:
            anchored.add(name)

    if seed is not None:
        seed_resolved = seed.resolve()
        matched = next((n for n, p in portrait_paths.items() if p == seed_resolved), None)
        if matched:
            anchored.add(matched)
        elif not anchored:
            anchored.update(names)
    return anchored


def panel_context(panel: dict, location_desc: str, key_props: list | None = None,
                  casting_lookup: dict[str, dict] | None = None) -> str:
    """[Context] — environment and background elements: the scene's locked
    location (from casting.json's location entry), this panel's own
    composition note (who/what is where in frame, depth layers), and the
    scene's key props. Props are part of the environment a Veo prompt has
    to assert explicitly, since nothing else in the five-part formula
    mentions them. Applied scene-wide (every panel in the scene gets the
    same prop list) rather than attributed to one specific panel, since no
    upstream artifact currently says which panel a given prop appears in.

    Each prop name is resolved against `casting_lookup` (same dict
    `panel_subject` already uses for characters) for a locked, very
    descriptive `visual_prompt` when one exists — a prop cast by
    `casting.py` (kind: "prop", recurring across 2+ scenes). This is what
    makes a recurring prop render as the SAME object across separately-
    generated clips. A prop mentioned only once in the story was never
    cast, so it falls back to its bare name here — no locked description
    exists for it."""
    bits = []
    if location_desc:
        bits.append(location_desc.rstrip("."))
    composition = (panel.get("composition") or "").strip()
    if composition:
        bits.append(composition.rstrip("."))
    casting_lookup = casting_lookup or {}
    prop_descs = []
    for p in (key_props or []):
        if not isinstance(p, str) or not p.strip():
            continue
        name = p.strip()
        entry = casting_lookup.get(name)
        desc = ""
        if entry and entry.get("kind") == "prop":
            desc = (entry.get("character", entry).get("visual_prompt") or "").strip()
        prop_descs.append(desc.rstrip(".") if desc else name)
    if prop_descs:
        bits.append("visible in the scene: " + "; ".join(prop_descs))
    return "; ".join(bits)


def panel_style_ambiance(visual_overview: dict) -> str:
    """[Style & Ambiance] — overall aesthetic, mood, and lighting: a fixed
    Veo style keyword plus the scene's visual_overview (color_palette/
    lighting_setup/mood)."""
    vo = visual_overview or {}
    bits = ["Cinematic, photorealistic"]
    bits += [b.strip() for b in
            (vo.get("color_palette", ""), vo.get("lighting_setup", ""), vo.get("mood", ""))
            if b.strip()]
    return ". ".join(b.rstrip(".") for b in bits if b) + "."


def five_part_veo_prompt(panel: dict, *, casting_lookup: dict[str, dict],
                         location_desc: str, visual_overview: dict,
                         anchored: set[str] | None = None) -> str:
    """Assemble the VISUAL half of a Veo prompt as an explicit, fixed-order
    dict, per the five-part formula:
      [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]
    Each section is built fresh from structured data (casting.json, the
    scene's visual_overview, the panel's own fields) rather than any agent's
    free-text prose — prose has no guaranteed internal ordering, so
    reordering an opaque blob after the fact isn't reliable; reconstructing
    from structured fields is.

    `anchored` — in-frame character names already covered by an actual
    seed/reference image THIS call (see `anchored_character_names`); passed
    straight through to `panel_subject`, which shortens their Subject text
    instead of restating what the image already shows."""
    parts: dict[str, str] = {
        "cinematography": panel_cinematography(panel),
        "subject": panel_subject(panel, casting_lookup, anchored),
        "action": (panel.get("action") or panel.get("moment") or "").strip(),
        "context": panel_context(panel, location_desc, (visual_overview or {}).get("key_props"),
                                 casting_lookup),
        "style_ambiance": panel_style_ambiance(visual_overview),
    }
    return " ".join(f"{v.rstrip('.')}." for v in parts.values() if v.strip())


def panel_relevant_characters(fr: dict) -> list[str]:
    """Which of this panel's `characters_in_frame` are actually relevant to
    THIS panel specifically, for the purpose of picking Veo reference/seed
    images. `characters_in_frame` is often a heuristic upstream (see
    `storyboard._panel_characters_in_frame`'s own docstring) — a wide/
    establishing/full shot (or a close shot with no dialogue) can default to
    the ENTIRE scene cast, not just whoever this specific panel's action
    actually depicts. Feeding that whole list into reference-image
    selection means a wide shot could burn Veo's 3-reference budget (or the
    single-seed slot) on a character technically in the scene but absent
    from this panel's own description — wasted budget at best, a wrong
    identity-lock at worst.

    A name counts as relevant here when it appears (word-boundary,
    case-insensitive) in the panel's own action/moment text, or is a
    dialogue speaker for this panel. Falls back to the FULL
    `characters_in_frame` list when nothing matches — a genuine group/
    establishing shot ("The crowd gathers.") names no one individually, and
    losing every reference image in that case would be worse than the
    over-inclusion this is narrowing. Does NOT affect `characters_in_frame`
    itself or anything else that reads it (Subject text, boundary/continuity
    tracking) — scoped narrowly to reference/seed image selection only."""
    names = list(dict.fromkeys(fr.get("characters_in_frame") or []))
    if not names:
        return names
    text = " ".join(b for b in ((fr.get("action") or ""), (fr.get("moment") or "")) if b)
    speakers = {(d.get("speaker") or "").strip() for d in (fr.get("dialogue") or [])}
    relevant = [name for name in names
               if name in speakers or re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE)]
    return relevant if relevant else names


def panel_video_prompt(panel: dict, audio_overview: dict | None = None, *,
                       casting_lookup: dict[str, dict] | None = None,
                       location_desc: str = "",
                       visual_overview: dict | None = None,
                       voice_index: dict[str, str] | None = None,
                       no_bg_music: bool = True,
                       room_tone: bool = True,
                       no_subtitles: bool = True,
                       out: Path | None = None,
                       seed: Path | None = None,
                       reference_images: list[Path] | None = None) -> str:
    """Assemble a complete Veo-aligned prompt (visual + audio halves) from a
    storyboard panel.

    `out`/`seed`/`reference_images` — the image(s), if any, actually being
    sent alongside this text in the same Veo call. When supplied (with
    `out`, required to resolve casting.json's relative `image_path`s), the
    Subject element shortens to a referential mention for any character
    already covered by one of those images instead of restating their full
    locked description in words — see `anchored_character_names`. Omitted
    by callers with no image context (e.g. a bare prompt build with no
    render happening), in which case Subject always writes the full text.

    Visual half ALWAYS follows the fixed five-part formula when
    `casting_lookup` is supplied — see `five_part_veo_prompt`. Callers
    without it (e.g. a bare prompt with no casting/scene context) fall back
    to the panel's own free-text `image_prompt`/`action`/`moment` instead —
    that text has no guaranteed section order, so the fixed formula is only
    guaranteed when `casting_lookup` is supplied.

    Audio half is built entirely via `veo_guide`'s construction helpers
    (`ambient_cue`/`sfx_cue`/`music_directive`/`dialogue_cue`/
    `no_subtitles_directive`) — that module owns the guide-vocabulary
    knowledge (which labels the guide actually confirms vs. which cases have
    no official keyword and need an unambiguous sentence instead), so this
    function only gathers the panel/scene data and hands it off. A future
    non-Veo backend can supply an equivalent module with the same call shape
    without touching this assembly logic.

    Veo generates audio per clip independently, with no memory of prior
    clips, so leaving any of these implicit invites drift across a scene:
    voice-over dialogue that gets lip-synced to an on-screen face, a
    hallucinated score that wasn't there before, room tone that suddenly
    gains an echo, or burned-in subtitle text. Spelling each one out
    explicitly — even the negative "no background music" / "no subtitles"
    cases — is what keeps the audio track consistent shot to shot.

    voice_index — character name → vocal-quality description (from characters.voice),
    so the same character sounds the same across separately-generated clips (Veo
    has no cross-generation voice cloning; this is the manual substitute)."""
    voice_index = voice_index or {}
    ao = audio_overview or {}

    # Ambient (environment) and score/music are kept as two SEPARATE cues (not
    # merged) — folding a score cue into "ambient" makes both drift together
    # when Veo hallucinates; keeping them apart lets the no-music directive
    # below cleanly override just the music half.
    panel_sound_raw = (panel.get("sound") or "").strip()
    if " | " in panel_sound_raw:
        panel_ambient, panel_sfx = [s.strip() for s in panel_sound_raw.split(" | ", 1)]
    else:
        panel_ambient, panel_sfx = "", panel_sound_raw

    ambient = panel_ambient or ao.get("ambient", "")
    sfx = panel_sfx
    if sfx and ambient and sfx.strip(".") == ambient.strip("."):
        sfx = ""

    # Anchor the room tone to consistent acoustics so it doesn't drift between
    # clips of the same scene (e.g. a sudden echo that wasn't there a shot ago).
    if room_tone and ambient and not any(
        t in ambient.lower() for t in ("echo", "reverb", "acoustic", "muffled", "hollow")
    ):
        ambient = f"{ambient.rstrip('.')}, dry acoustics, no echo"

    score = (ao.get("score_cue") or "").strip()

    dialogue_cues: list[str] = []
    for d in (panel.get("dialogue") or []):
        line = (d.get("line") or "").strip()
        if not line:
            continue
        modifier = (d.get("modifier") or "").strip().upper()
        speaker = (d.get("speaker") or "").strip()
        cue = veo_guide.dialogue_cue(
            speaker, line,
            voice=voice_index.get(speaker, ""),
            tone=(d.get("parenthetical") or "").strip().strip("()"),
            vo=bool(d.get("vo")),
            off_screen="O.S." in modifier,
        )
        if cue:
            dialogue_cues.append(cue)

    # Visual half — always the fixed five-part formula when casting_lookup is
    # available (the real render path always supplies it); otherwise fall
    # back to the panel's own free-text image_prompt.
    if casting_lookup is not None:
        anchored = (anchored_character_names(
                        panel.get("characters_in_frame") or [], casting_lookup,
                        out, seed, reference_images)
                    if out is not None else None)
        visual = five_part_veo_prompt(panel, casting_lookup=casting_lookup,
                                      location_desc=location_desc,
                                      visual_overview=visual_overview or {},
                                      anchored=anchored)
    else:
        visual = (panel.get("image_prompt") or panel.get("action")
                 or panel.get("moment", "")).strip()

    # Audio half: ambient → SFX → music/no-music → dialogue (+ no-subtitles).
    audio_bits: list[str] = []
    for cue in (veo_guide.ambient_cue(ambient), veo_guide.sfx_cue(sfx),
               veo_guide.music_directive(score, no_background_music=no_bg_music)):
        if cue:
            audio_bits.append(cue)
    audio_bits.extend(dialogue_cues)
    if dialogue_cues and no_subtitles:
        audio_bits.append(veo_guide.no_subtitles_directive())

    return (visual + (" " + " ".join(audio_bits) if audio_bits else "")).strip()


def panel_dialogue_lines(panel: dict) -> list[str]:
    """Format dialogue entries from a storyboard panel as subtitle strings."""
    lines = []
    for d in (panel.get("dialogue") or []):
        speaker = (d.get("speaker") or "").strip()
        line = (d.get("line") or "").strip()
        if speaker and line:
            lines.append(f"{speaker}: {line}")
        elif line:
            lines.append(line)
    return lines
