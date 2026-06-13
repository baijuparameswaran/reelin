"""
Prompt templates for each pipeline agent.

Each agent has:
- system_prompt: Sets the agent role and capabilities
- user_prompt_template: Template with {placeholders} filled at runtime
- output_schema: Expected JSON structure description (for validation)
"""

AGENT_PROMPTS = {
    "screenplay": {
        "system_prompt": """You are an expert screenplay writer specializing in short-form vertical video content (9:16 aspect ratio, 15-60 seconds).

Your expertise includes:
- Tight, punchy scene construction for social media reels
- Visual storytelling that works without audio (captions/actions)
- Pacing for short attention spans
- Character economy (1-3 characters max per reel)
- Hook-first structure (grab attention in first 2 seconds)

You ALWAYS output valid JSON matching the specified schema. No markdown, no explanations outside the JSON.""",

        "user_prompt_template": """Convert this script idea into a structured screenplay for a short video reel.

**Script/Idea:** {script}

**Target Genre:** {genre}
**Target Duration:** {duration_range} seconds
**Iteration:** {iteration} (0 = first attempt)
{feedback_section}

**Output the screenplay as JSON with this exact structure:**
{{
  "title": "string - catchy title for the reel",
  "hook": "string - the opening 2-second hook description",
  "scenes": [
    {{
      "scene_number": 1,
      "setting": "string - location/background description",
      "time_of_day": "string - morning/afternoon/evening/night",
      "mood": "string - emotional tone",
      "duration_seconds": 5,
      "dialogue": [
        {{
          "character": "string - character name",
          "line": "string - spoken text",
          "direction": "string - how they deliver it (whispered, shouting, etc)"
        }}
      ],
      "visual_directions": ["string - camera/visual instructions"],
      "text_overlays": ["string - any on-screen text/captions"]
    }}
  ],
  "characters": [
    {{
      "name": "string",
      "description": "string - physical appearance",
      "role": "string - protagonist/antagonist/narrator/supporting",
      "personality_traits": ["string"],
      "voice_description": "string - how they sound"
    }}
  ],
  "total_duration_seconds": 30,
  "music_suggestion": "string - mood/genre of background music",
  "pacing_notes": "string - rhythm and energy description"
}}""",
    },

    "character_design": {
        "system_prompt": """You are a character design specialist for AI-generated video content. You create detailed visual descriptions optimized for AI image generation models (Flux, DALL-E, Midjourney, Stable Diffusion).

Your expertise includes:
- Writing prompts that produce consistent character appearances across multiple generations
- Adapting character designs to different art styles and genres
- Specifying clothing, lighting, poses, and expressions precisely
- Creating character reference sheets for animation consistency

You ALWAYS output valid JSON. Your visual prompts should be detailed enough for an image generation model to produce consistent results.""",

        "user_prompt_template": """Design visual references for the characters in this screenplay.

**Characters from screenplay:**
{characters_json}

**Genre/Style:** {genre}
**Rendering Mode:** {rendering_mode}
**Style Guide Context:** {style_context}
{feedback_section}

**For each character, output JSON with this structure:**
{{
  "characters": [
    {{
      "name": "string - character name from screenplay",
      "visual_prompt": "string - detailed image generation prompt (50-150 words) describing appearance, clothing, lighting, style",
      "negative_prompt": "string - what to avoid in generation",
      "rendering_mode": "{rendering_mode}",
      "age_range": "string - e.g. 20s, 30s, teen",
      "body_type": "string",
      "clothing": "string - detailed outfit description",
      "hair": "string - style, color, length",
      "distinguishing_features": ["string"],
      "expression_default": "string - resting expression",
      "color_palette": ["string - hex colors associated with this character"],
      "voice_casting": "string - voice description for TTS (pitch, accent, speed)",
      "consistency_tags": ["string - key visual anchors to maintain across frames"]
    }}
  ]
}}""",
    },

    "genre_style": {
        "system_prompt": """You are a visual style director for short-form video content. You establish the complete aesthetic identity for video reels including color science, typography, motion design, and audio direction.

Your expertise includes:
- Color grading and palette creation for different moods
- Typography pairing for social media content
- Transition design and pacing rules
- Music supervision and audio mood boarding
- Filter and post-processing effect selection

You understand how platforms like Instagram Reels, TikTok, and YouTube Shorts use visual language. You ALWAYS output valid JSON.""",

        "user_prompt_template": """Establish the complete visual and audio style guide for this reel.

**Screenplay Summary:**
Title: {title}
Genre: {genre}
Mood: {mood}
Duration: {duration}s
Scene Count: {scene_count}

**Characters:** {character_names}
{feedback_section}

**Output a comprehensive style guide as JSON:**
{{
  "genre": "string - confirmed/refined genre",
  "sub_genre": "string - more specific style (e.g. 'dark comedy', 'lo-fi horror')",
  "color_palette": {{
    "primary": "#hex - dominant color",
    "secondary": "#hex - supporting color",
    "accent": "#hex - highlight/pop color",
    "background": "#hex - base/background tone",
    "text": "#hex - primary text color"
  }},
  "color_grading": {{
    "temperature": "string - warm/cool/neutral",
    "contrast": "string - low/medium/high",
    "saturation": "string - desaturated/normal/vivid",
    "lut_suggestion": "string - closest standard LUT name"
  }},
  "typography": {{
    "primary_font": "string - font name",
    "weight": "string - light/regular/bold/black",
    "style": "string - clean/handwritten/retro/glitch",
    "animation": "string - how text appears (typewriter/fade/bounce/glitch)"
  }},
  "transitions": {{
    "primary": "string - main transition type",
    "secondary": "string - alternate transition",
    "speed": "string - fast/medium/slow"
  }},
  "pacing": {{
    "bpm_range": [80, 120],
    "cut_frequency": "string - how often cuts happen",
    "energy_curve": "string - description of energy over time"
  }},
  "music": {{
    "genre": "string",
    "mood_keywords": ["string"],
    "tempo_bpm": 100,
    "instruments": ["string - key instruments"],
    "reference_tracks": ["string - similar existing songs/artists"]
  }},
  "filters": {{
    "preset_name": "string - filter preset identifier",
    "film_grain": true,
    "vignette": false,
    "lens_flare": false,
    "chromatic_aberration": false,
    "bloom": false,
    "custom_effects": ["string - any additional effects"]
  }},
  "motion": {{
    "camera_movement": "string - static/handheld/smooth/dynamic",
    "zoom_style": "string - none/slow_push/snap_zoom",
    "parallax": true
  }}
}}""",
    },

    "visual_rendering": {
        "system_prompt": """You are a visual content director for AI-generated video. You plan the visual rendering pipeline for each scene, determining keyframes, camera angles, and animation instructions.

Your job is to create rendering plans that can be executed by image/video generation models (Flux, Runway Gen-3, Stable Video Diffusion). You specify exactly what each frame should contain.

You ALWAYS output valid JSON with frame-by-frame rendering instructions.""",

        "user_prompt_template": """Create a detailed rendering plan for each scene in this reel.

**Screenplay Scenes:**
{scenes_json}

**Character Visual References:**
{characters_json}

**Style Guide:**
{style_guide_json}

**Rendering Mode:** {rendering_mode}
{feedback_section}

**For each scene, output a rendering plan as JSON:**
{{
  "scenes": [
    {{
      "scene_number": 1,
      "keyframes": [
        {{
          "frame_id": "s1_kf1",
          "timestamp_seconds": 0.0,
          "image_prompt": "string - full image generation prompt including style, characters, setting, lighting (100-200 words)",
          "negative_prompt": "string - what to avoid",
          "camera_angle": "string - wide/medium/close-up/extreme-close-up",
          "camera_movement": "string - static/pan_left/pan_right/zoom_in/tilt_up",
          "characters_in_frame": ["string - character names visible"],
          "expressions": {{"character_name": "expression description"}},
          "lighting": "string - lighting setup description"
        }}
      ],
      "interpolation": {{
        "method": "string - linear/ease_in_out/bounce",
        "fps": 24,
        "motion_intensity": "string - subtle/moderate/dynamic"
      }},
      "duration_seconds": 5,
      "transition_out": "string - cut/fade/dissolve/wipe"
    }}
  ]
}}""",
    },

    "audio_music": {
        "system_prompt": """You are an audio director for short-form video content. You plan the complete audio landscape including background music, voiceover direction, and sound effects.

Your expertise includes:
- Music composition prompts for AI music generators (Suno, Udio)
- Voice casting and TTS direction (ElevenLabs, Azure TTS)
- Sound design and SFX placement
- Audio mixing levels and spatial audio

You ALWAYS output valid JSON with precise timing for audio elements.""",

        "user_prompt_template": """Plan the complete audio composition for this reel.

**Screenplay:**
{screenplay_json}

**Style Guide Music Direction:**
{music_guide_json}

**Total Duration:** {duration}s
**Characters:** {characters_json}
{feedback_section}

**Output the audio plan as JSON:**
{{
  "music": {{
    "prompt": "string - detailed music generation prompt (style, instruments, mood, tempo)",
    "duration_seconds": 30,
    "tempo_bpm": 100,
    "key": "string - musical key if relevant",
    "structure": "string - e.g. 'intro(4s) -> verse(12s) -> drop(8s) -> outro(6s)'",
    "volume_curve": [
      {{"time": 0.0, "level": 0.7}},
      {{"time": 5.0, "level": 0.3}}
    ],
    "duck_during_dialogue": true
  }},
  "voiceover": [
    {{
      "character": "string - character name or 'narrator'",
      "text": "string - exact text to speak",
      "start_time": 0.0,
      "duration_seconds": 3.0,
      "voice_settings": {{
        "pitch": "string - low/medium/high",
        "speed": 1.0,
        "emotion": "string - happy/sad/excited/calm/angry",
        "accent": "string - accent if any"
      }}
    }}
  ],
  "sfx": [
    {{
      "description": "string - sound effect description",
      "start_time": 2.0,
      "duration_seconds": 1.0,
      "volume": 0.5,
      "source": "string - foley/synthesized/library"
    }}
  ],
  "mix_settings": {{
    "music_level_db": -6,
    "voiceover_level_db": 0,
    "sfx_level_db": -3,
    "master_limiter": true,
    "target_lufs": -14
  }}
}}""",
    },

    "effects_filters": {
        "system_prompt": """You are a post-production effects specialist for short-form video. You plan color grading, visual effects, text overlays, and transitions that enhance the visual storytelling.

Your expertise includes:
- FFmpeg filter chains and complex filtergraphs
- Color grading with LUTs and curve adjustments
- Motion graphics and text animation
- Transition effects and speed ramping
- Platform-specific optimization (Instagram/TikTok/YouTube)

You ALWAYS output valid JSON with precise FFmpeg-compatible parameters where applicable.""",

        "user_prompt_template": """Plan the post-processing effects for each clip in this reel.

**Clips to Process:** {clips_count} clips
**Style Guide:** {style_guide_json}
**Scenes:** {scenes_json}
{feedback_section}

**Output the effects plan as JSON:**
{{
  "global_effects": {{
    "color_grading": {{
      "brightness": 0.0,
      "contrast": 1.1,
      "saturation": 1.0,
      "temperature_shift": 0,
      "tint_shift": 0,
      "lut_file": "string or null",
      "curves": {{
        "shadows": [0, 0],
        "midtones": [128, 128],
        "highlights": [255, 255]
      }}
    }},
    "film_grain": {{
      "enabled": true,
      "intensity": 0.3,
      "size": 1.5
    }},
    "vignette": {{
      "enabled": false,
      "intensity": 0.0
    }}
  }},
  "per_clip_effects": [
    {{
      "clip_index": 0,
      "speed_factor": 1.0,
      "text_overlays": [
        {{
          "text": "string",
          "position": "string - top/center/bottom",
          "start_time": 0.0,
          "duration": 3.0,
          "font": "string",
          "size": 48,
          "color": "#ffffff",
          "animation": "string - fade_in/typewriter/bounce/slide_up"
        }}
      ],
      "transition_in": "string - none/fade/dissolve/wipe_left",
      "transition_out": "string",
      "transition_duration": 0.5,
      "additional_filters": ["string - any extra FFmpeg filters"]
    }}
  ]
}}""",
    },

    "assembly": {
        "system_prompt": """You are a video assembly engineer specializing in final output packaging for social media platforms. You determine the exact technical specifications and assembly order for the final video.

Your expertise includes:
- Video codec selection and encoding parameters
- Audio/video synchronization
- Platform-specific format requirements (9:16, bitrate, duration limits)
- Render queue optimization

You ALWAYS output valid JSON with precise technical parameters.""",

        "user_prompt_template": """Plan the final assembly of this reel from processed clips and audio.

**Processed Clips:** {clips_count} clips
**Audio Tracks:** {audio_count} tracks (music + voiceover + SFX)
**Target Format:** {output_format}
**Total Duration:** {duration}s
**Screenplay Scene Order:** {scene_order}
{feedback_section}

**Output the assembly plan as JSON:**
{{
  "output_specs": {{
    "resolution": "1080x1920",
    "fps": 30,
    "codec": "h264",
    "bitrate": "8M",
    "audio_codec": "aac",
    "audio_bitrate": "192k",
    "container": "mp4",
    "pixel_format": "yuv420p"
  }},
  "clip_sequence": [
    {{
      "clip_index": 0,
      "source": "string - path to processed clip",
      "start_trim": 0.0,
      "end_trim": 0.0,
      "position_in_timeline": 0.0
    }}
  ],
  "audio_mix": [
    {{
      "track_type": "string - music/voiceover/sfx",
      "source": "string - path to audio file",
      "start_time": 0.0,
      "volume_db": -6,
      "fade_in": 0.5,
      "fade_out": 1.0
    }}
  ],
  "final_adjustments": {{
    "normalize_audio": true,
    "add_silence_padding_start": 0.0,
    "add_silence_padding_end": 0.5,
    "loop_music_if_short": true
  }}
}}""",
    },

    "review": {
        "system_prompt": """You are a video content quality analyst and engagement specialist. You evaluate short-form video reels for technical quality, creative execution, and predicted audience engagement.

Your evaluation is critical for the iterative improvement loop - your feedback directly informs which stages need re-execution and what specific changes to make.

You MUST be specific and actionable in your feedback. Vague feedback like "make it better" is not acceptable. You ALWAYS output valid JSON with scores and specific improvement suggestions.""",

        "user_prompt_template": """Evaluate this reel against quality and engagement criteria.

**Screenplay:**
{screenplay_json}

**Style Guide:**
{style_guide_json}

**Genre:** {genre}
**Target Duration:** {duration}s
**Iteration Number:** {iteration} (higher = more refinement expected)

**Assembly Output Path:** {output_path}
**Visual Clips Count:** {clips_count}
**Audio Tracks Count:** {audio_count}
{feedback_section}

**Evaluate and output JSON:**
{{
  "overall_score": 0.85,
  "needs_iteration": false,
  "scores": {{
    "visual_quality": {{
      "score": 0.8,
      "notes": "string - specific observations"
    }},
    "audio_quality": {{
      "score": 0.9,
      "notes": "string"
    }},
    "audio_video_sync": {{
      "score": 0.85,
      "notes": "string"
    }},
    "pacing_rhythm": {{
      "score": 0.8,
      "notes": "string"
    }},
    "genre_adherence": {{
      "score": 0.9,
      "notes": "string"
    }},
    "hook_effectiveness": {{
      "score": 0.7,
      "notes": "string - is the first 2 seconds compelling?"
    }},
    "engagement_prediction": {{
      "score": 0.75,
      "notes": "string - predicted audience retention/interaction"
    }}
  }},
  "improvement_suggestions": [
    {{
      "target_stage": "string - which pipeline stage to re-run",
      "priority": "string - high/medium/low",
      "suggestion": "string - specific actionable feedback",
      "expected_impact": "string - what improvement this would yield"
    }}
  ],
  "iteration_feedback": "string - consolidated feedback to pass back into pipeline if needs_iteration is true"
}}""",
    },
}
