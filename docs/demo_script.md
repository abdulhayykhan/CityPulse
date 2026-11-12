# CityPulse — Demo Script

A rehearsed ~3-minute walkthrough for the live demo and the recorded video.
Rehearse it twice before recording (Part 5.6 of the execution guide).

## Before you start
1. `conda activate citypulse` then `.\run.ps1` (or `./run.sh`). Open **http://localhost:8000/**.
2. Make sure the Brooklyn drive graph is cached (click **Plan route** once beforehand so the
   first OSM download doesn't happen on camera).
3. Optional clean slate: stop the server, delete `data/citypulse.db`, restart — this clears
   previous demo pins so the map starts tidy.

## Walkthrough

1. **Open the dashboard.** Point out the real hotspot **zones** built from public NYC 311
   complaints, coloured by livability (green = more livable, red = problem-heavy), and the
   incident markers coloured by type. Note the stat strip (signals, zones, avg livability).

2. **Submit a held-out audio clip.** In *Submit a report*, keep **Audio**, pick the
   `construction — …jackhammer` sample, click a spot on the map, then **Analyze & drop pin**.
   A construction pin appears; read out the **confidence** and **relative-intensity** meters.

3. **Submit a held-out photo.** Switch to **Image**, pick `pothole_01.jpg`, click a different
   spot, **Analyze**. A second pin drops with its confidence — two modalities, one pipeline.

4. **Show activity building up.** Drop two or three more reports near one location to show how
   repeated reports cluster around a hotspot (the offline pipeline merges same-place,
   same-type reports within 200 m / 4 h into single incidents).

5. **Move the rain slider.** Set *Map coloring* to **Flood risk** and push **Rainfall** up —
   the zones recolour as `predict_flood_risk` raises expected flood complaints. Switch to
   **Report density** to show low-data areas are flagged, never shown as "safe".

6. **Plan a calm & safe route.** Click **Set points**, click a start and an end, **Plan route**.
   Show the plain route (grey dashed) and the safe route (teal) and the detour distance in the
   toast. Then drop a report **on the safe route** and re-plan — the safe route **detours
   around the new hazard**. This is the headline moment.

7. **Close with the honest caveat.** State plainly what the data is and is not: everything is
   trained and computed on **public datasets** (UrbanSound8K, ESC-50, RDD2022, NYC 311,
   Open-Meteo, OpenStreetMap) — **not Karachi field measurements**. Degradation and
   cross-dataset results are reported honestly; trust and zone rankings are proxy signals.

## Recording tips
- Windows: **Win + G** (Xbox Game Bar) or OBS Studio to capture the browser window; Loom also works.
- Keep it 2–3 minutes, 1080p, browser maximised so the right-hand insight panel is visible.
- Do a silent dry run first, then record with narration following the seven steps above.
