# fix_aoi.py
import json

geo = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [77.55, 12.90],
            [77.65, 12.90],
            [77.65, 12.95],
            [77.55, 12.95],
            [77.55, 12.90]
          ]
        ]
      }
    }
  ]
}

with open("aoi.geojson", "w", encoding="utf-8") as f:
    json.dump(geo, f, ensure_ascii=False, indent=2)

print("✅ Wrote valid aoi.geojson to project root.")
