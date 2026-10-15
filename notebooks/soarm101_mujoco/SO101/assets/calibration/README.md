# Table calibration sheets

`sheet_<name>.png`: bambot's printable table calibration sheets
(`bambot/public/calibration/table_calibration_a4*.svg`) rasterised at 150 dpi
(A4, 1239 x 1753 px, greyscale), for `SO101/scene.xml`. The scene lays them out
as `table_calibration_a4.json` does: page top towards base_link +x, page left
towards +y, on the table at `table_z_mm`.

They were rendered with headless Chromium (page opened at the SVG, device
scale 150/96, screenshot of the 793.7 x 1122.5 css px page). Re-render them
when the sheets change; if the layout changes, update the sheet geoms in
`scene.xml` from the JSON too.
