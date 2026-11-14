"""Interactive 3D Earth (globe.gl / three.js) with our computed results on it.

Hexagonal columns = computed risk aggregated on an H3 grid (height and colour = mean risk),
outlines = real NISAR GUNW frames from the ASF search, dots = detected events and known
landslide sites. Hover shows the calculated values; clicking the study area flies in.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np

GLOBE_JS = "https://cdn.jsdelivr.net/npm/globe.gl@2.44.0/dist/globe.gl.min.js"
TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile"
IMG = "https://cdn.jsdelivr.net/npm/three-globe@2.31.0/example/img/"


def png_data_uri(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colorize(arr, vmin, vmax, stops):
    """Map array to RGB with linear interpolation between colour stops [(pos, (r,g,b)), ...]."""
    t = np.clip((np.asarray(arr, dtype=np.float64) - vmin) / (vmax - vmin + 1e-12), 0, 1)
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops], dtype=np.float64)
    out = np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], -1)
    return out.astype(np.uint8)


RISK_STOPS = [(0.0, (34, 49, 63)), (0.3, (46, 160, 110)), (0.6, (240, 190, 40)),
              (0.8, (232, 110, 40)), (1.0, (214, 40, 40))]


def build_globe_html(risk_map, bounds, events, footprints, sites, thumb_uri, stats: dict,
                     height: int = 620, step: int = 6) -> str:
    lat_min, lat_max, lon_min, lon_max = bounds
    h, w = risk_map.shape
    rr, cc = np.mgrid[0:h:step, 0:w:step]
    lat = lat_max - rr / (h - 1) * (lat_max - lat_min)
    lon = lon_min + cc / (w - 1) * (lon_max - lon_min)
    pts = [{"lat": round(float(a), 4), "lng": round(float(b), 4), "w": round(float(v), 1)}
           for a, b, v in zip(lat.ravel(), lon.ravel(), risk_map[rr, cc].ravel())]

    ev = [{"lat": f["properties"]["lat"], "lng": f["properties"]["lon"],
           "id": f["properties"]["event_id"], "level": f["properties"]["risk_level"],
           "score": f["properties"]["risk_score"],
           "disp": f["properties"].get("max_subsidence_mm"),
           "area": f["properties"].get("area_km2")}
          for f in events.get("features", []) if "lat" in f["properties"]]
    centre = {"lat": (lat_min + lat_max) / 2, "lng": (lon_min + lon_max) / 2}
    aoi = [[lat_min, lon_min], [lat_min, lon_max], [lat_max, lon_max], [lat_max, lon_min],
           [lat_min, lon_min]]

    payload = json.dumps(dict(pts=pts, ev=ev, fps=footprints, sites=sites, centre=centre,
                              aoi=aoi, thumb=thumb_uri, stats=stats))
    return f"""
<div id="wrap" style="position:relative;height:{height}px;background:#05070c;border-radius:6px;overflow:hidden">
  <div id="globe"></div>
  <div id="panel" style="position:absolute;left:14px;top:12px;max-width:290px;background:rgba(5,7,12,.72);padding:8px 10px;border-radius:5px;font:13px 'IBM Plex Sans',system-ui,sans-serif;color:#cfd6df;pointer-events:none">
    <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#8a96a5">Computed from NISAR L-band InSAR</div>
    <div style="margin-top:4px">Hex height &amp; colour = mean risk. Hover a hex, event or frame for the numbers. Click the study area to fly in.</div>
  </div>
  <div style="position:absolute;right:14px;bottom:12px;background:rgba(5,7,12,.72);padding:4px 8px;border-radius:4px;font:12px 'IBM Plex Sans',system-ui,sans-serif;color:#8a96a5">
    Imagery © Esri · <span style="color:#2ea06e">■</span> low &nbsp;<span style="color:#f0be28">■</span> medium &nbsp;<span style="color:#e86e28">■</span> high &nbsp;<span style="color:#d62828">■</span> critical &nbsp; <span style="color:#6cb4ff">▭</span> NISAR frames
  </div>
</div>
<script src="{GLOBE_JS}"></script>
<script>
const D = {payload};
const RISK = [[0,[34,49,63]],[30,[46,160,110]],[60,[240,190,40]],[80,[232,110,40]],[100,[214,40,40]]];
function riskColor(v, a=1) {{
  for (let i=1;i<RISK.length;i++) if (v<=RISK[i][0]) {{
    const [p0,c0]=RISK[i-1],[p1,c1]=RISK[i],t=(v-p0)/(p1-p0);
    const c=c0.map((x,k)=>Math.round(x+(c1[k]-x)*t)); return `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{a}})`;
  }}
  return `rgba(214,40,40,${{a}})`;
}}
const LVL = {{LOW:'#2ea06e',MEDIUM:'#f0be28',HIGH:'#e86e28',CRITICAL:'#d62828'}};
const card = (html) => `<div style="background:rgba(10,14,22,.92);border:1px solid #2a3442;border-radius:6px;padding:8px 10px;font:12px 'IBM Plex Sans',system-ui,sans-serif;color:#dde3ea;max-width:260px">${{html}}</div>`;
const S = D.stats;
const aoiCard = card(`<b>Almaty study area</b><br>${{S.source}}<br>
  <img src="${{D.thumb}}" style="width:230px;margin:6px 0;border-radius:3px;display:block">
  Events: <b>${{S.n_events}}</b> · overall risk <b>${{S.level}}</b> (${{S.overall}})<br>
  Max subsidence: <b>${{S.max_sub}} mm</b> · area ${{S.area}} km²`);

const el = document.getElementById('globe');
const g = Globe()(el)
  .width(el.parentElement.clientWidth).height({height})
  .globeImageUrl('{IMG}earth-blue-marble.jpg')
  .globeTileEngineUrl((x, y, l) => `{TILES}/${{l}}/${{y}}/${{x}}`)
  .backgroundImageUrl('{IMG}night-sky.png')
  .atmosphereColor('#6cb4ff').atmosphereAltitude(0.18)
  // computed risk, aggregated on H3 cells
  .hexBinPointsData(D.pts).hexBinPointLat('lat').hexBinPointLng('lng').hexBinPointWeight('w')
  .hexBinResolution(7).hexMargin(0.12).hexBinMerge(false)
  .hexAltitude(d => 0.0002 + (d.sumWeight / d.points.length) / 100 * 0.006)
  .hexTopColor(d => riskColor(d.sumWeight / d.points.length, 0.95))
  .hexSideColor(d => riskColor(d.sumWeight / d.points.length, 0.55))
  .hexLabel(d => {{
    const m = d.sumWeight / d.points.length, p = d.points[0];
    return card(`Risk <b>${{m.toFixed(1)}}</b> / 100<br>${{p.lat.toFixed(3)}}°N, ${{p.lng.toFixed(3)}}°E<br><span style="color:#8a96a5">${{d.points.length}} pixels averaged</span>`);
  }})
  // real NISAR frame outlines + the study area
  .pathsData([...D.fps.map(f => ({{...f, kind:'frame'}})), {{coords: D.aoi.map(([a,b]) => [b,a]), kind:'aoi'}}])
  .pathPoints(d => d.coords).pathPointLat(p => p[1]).pathPointLng(p => p[0])
  .pathColor(d => d.kind === 'aoi' ? '#ffffff' : 'rgba(108,180,255,0.75)')
  .pathStroke(d => d.kind === 'aoi' ? 1.6 : 1.0)
  .pathDashLength(d => d.kind === 'aoi' ? 0.02 : 1).pathDashGap(d => d.kind === 'aoi' ? 0.01 : 0)
  .pathLabel(d => d.kind === 'aoi' ? aoiCard : card(`<b>NISAR GUNW frame</b><br>Track ${{d.track}} · frame ${{d.frame}} · ${{d.direction}}<br>${{d.n}} pair(s) · latest ${{d.latest}}<br>AOI coverage ${{d.coverage}}%`))
  // detected events + known sites
  .pointsData([...D.ev.map(e => ({{...e, kind:'event'}})), ...D.sites.map(s => ({{...s, kind:'site'}}))])
  .pointLat('lat').pointLng('lng')
  .pointColor(d => d.kind === 'site' ? '#ffffff' : LVL[d.level])
  .pointAltitude(d => d.kind === 'site' ? 0.0035 : 0.0015 + d.score / 100 * 0.006)
  .pointRadius(d => d.kind === 'site' ? 0.006 : 0.009)
  .pointLabel(d => d.kind === 'site'
     ? card(`<b>${{d.name}}</b><br>${{d.note}}<br><span style="color:#8a96a5">reference site (approx. location)</span>`)
     : card(`<b>${{d.id}}</b> · <span style="color:${{LVL[d.level]}}">${{d.level}}</span> (${{d.score}})<br>Max subsidence ${{d.disp}} mm<br>Area ${{d.area}} km²`))
  .ringsData(D.ev.filter(e => e.level === 'HIGH' || e.level === 'CRITICAL'))
  .ringLat('lat').ringLng('lng').ringColor(d => t => `rgba(232,110,40,${{1-t}})`)
  .ringMaxRadius(0.06).ringPropagationSpeed(0.4).ringRepeatPeriod(1400)
  .labelsData([{{...D.centre, text:'Almaty'}}]).labelLat('lat').labelLng('lng').labelText('text')
  .labelSize(0.045).labelDotRadius(0.008).labelColor(() => '#ffffff').labelAltitude(0.01)
  .labelLabel(() => aoiCard)
  .onLabelClick(() => g.pointOfView({{lat: D.centre.lat - 0.3, lng: D.centre.lng, altitude: 0.05}}, 2200))
  .onPathClick(d => d.kind === 'aoi' && g.pointOfView({{lat: D.centre.lat - 0.3, lng: D.centre.lng, altitude: 0.05}}, 2200))
  .pointOfView({{lat: 40, lng: 72, altitude: 2.3}});

g.controls().autoRotate = true; g.controls().autoRotateSpeed = 0.35;
el.addEventListener('pointerdown', () => g.controls().autoRotate = false, {{once:true}});
setTimeout(() => {{ g.controls().autoRotate = false;
  g.pointOfView({{lat: D.centre.lat - 0.3, lng: D.centre.lng, altitude: 0.05}}, 4500); }}, 2000);
window.addEventListener('resize', () => g.width(el.parentElement.clientWidth));
</script>
"""
