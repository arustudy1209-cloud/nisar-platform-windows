"""Map and chart builders for the Streamlit UI."""

from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import pydeck as pdk

from .coherence import CoherenceAnalyzer


class DisasterMap:
    def __init__(self, map_style: str = pdk.map_styles.CARTO_DARK):
        self.map_style = map_style

    @staticmethod
    def _grid_lonlat(shape, bounds, step):
        lat_min, lat_max, lon_min, lon_max = bounds
        h, w = shape
        rows = np.arange(0, h, step)
        cols = np.arange(0, w, step)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        lat = lat_max - rr / (h - 1) * (lat_max - lat_min)
        lon = lon_min + cc / (w - 1) * (lon_max - lon_min)
        return rr, cc, lat, lon

    def create_3d_map(self, displacement_map, damage_mask, dem, lat_lon_bounds,
                      events: dict | None = None, heat_step: int = 6) -> pdk.Deck:
        """3D map: heatmap of |displacement|, columns on disaster zones, risk polygons."""
        lat_min, lat_max, lon_min, lon_max = lat_lon_bounds
        disp = np.asarray(displacement_map)
        shape = disp.shape

        # Heatmap of displacement magnitude
        rr, cc, lat, lon = self._grid_lonlat(shape, lat_lon_bounds, heat_step)
        mag = np.abs(disp[rr, cc])
        keep = mag > np.percentile(mag, 40)
        heat_data = [
            {"lon": float(a), "lat": float(b), "w": float(m)}
            for a, b, m in zip(lon[keep], lat[keep], mag[keep])
        ]
        heat = pdk.Layer(
            "HeatmapLayer",
            data=heat_data,
            get_position=["lon", "lat"],
            get_weight="w",
            radius_pixels=40,
            intensity=1.0,
            threshold=0.05,
            opacity=0.55,
            color_range=[
                [30, 60, 120], [50, 130, 190], [90, 200, 200],
                [250, 220, 90], [250, 140, 50], [230, 50, 50],
            ],
        )

        # 3D columns on damaged pixels (subsampled)
        col_step = max(2, heat_step // 2)
        rr2, cc2, lat2, lon2 = self._grid_lonlat(shape, lat_lon_bounds, col_step)
        dm = np.asarray(damage_mask)[rr2, cc2]
        d = disp[rr2, cc2]
        z = np.asarray(dem)[rr2, cc2]
        dmax = np.abs(d[dm]).max() if dm.any() else 1.0
        col_data = []
        for a, b, v, e in zip(lon2[dm], lat2[dm], d[dm], z[dm]):
            t = min(abs(v) / dmax, 1.0)
            col_data.append({
                "lon": float(a), "lat": float(b),
                "disp": round(float(v), 1), "dem": round(float(e)),
                "height": float(abs(v)),
                "color": [int(255), int(200 - 170 * t), int(60 - 40 * t), 220],
            })
        columns = pdk.Layer(
            "ColumnLayer",
            data=col_data,
            get_position=["lon", "lat"],
            get_elevation="height",
            elevation_scale=12,
            radius=120,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        layers = [heat, columns]

        # Risk-level polygons
        if events and events.get("features"):
            poly_data = []
            for f in events["features"]:
                p = f["properties"]
                rgb = CoherenceAnalyzer.level_color(p.get("risk_level", "LOW"))
                poly_data.append({
                    "polygon": f["geometry"]["coordinates"][0],
                    "fill": rgb + [60],
                    "line": rgb + [255],
                    "event_id": p.get("event_id"),
                    "risk_level": p.get("risk_level"),
                    "risk_score": p.get("risk_score"),
                    "disp": p.get("max_subsidence_mm"),
                    "dem": "",
                })
            layers.append(pdk.Layer(
                "PolygonLayer",
                data=poly_data,
                get_polygon="polygon",
                get_fill_color="fill",
                get_line_color="line",
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
                pickable=True,
            ))

        view = pdk.ViewState(
            latitude=(lat_min + lat_max) / 2,
            longitude=(lon_min + lon_max) / 2,
            zoom=10,
            pitch=45,
            bearing=0,
        )
        tooltip = {
            "html": "<b>{event_id}</b> {risk_level} {risk_score}<br/>"
                    "Displacement: <b>{disp}</b> mm<br/>Elevation: {dem} m",
            "style": {"backgroundColor": "#1a1a2e", "color": "white"},
        }
        return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto",
                        map_style=self.map_style, tooltip=tooltip)

    @staticmethod
    def create_displacement_timeline(dates, displacements: dict) -> go.Figure:
        """dates: list of dates; displacements: {series_name: [values_mm]}."""
        fig = go.Figure()
        palette = ["#FF4B4B", "#ff8c42", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
        for i, (name, vals) in enumerate(displacements.items()):
            fig.add_trace(go.Scatter(
                x=list(dates), y=list(vals), mode="lines+markers", name=name,
                line=dict(width=2.5, color=palette[i % len(palette)]),
                marker=dict(size=6),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} mm<extra>" + name + "</extra>",
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Cumulative LOS displacement per event",
            xaxis_title="Acquisition date",
            yaxis_title="Displacement (mm)",
            legend=dict(orientation="h", y=-0.2),
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    @staticmethod
    def create_risk_panel(events) -> str:
        """HTML table of events with colour-coded ● markers."""
        feats = events.get("features", []) if isinstance(events, dict) else events
        if not feats:
            return "<p style='color:#aaa'>No events detected.</p>"
        rows = []
        for f in feats:
            p = f["properties"] if "properties" in f else f
            lvl = p.get("risk_level", "LOW")
            rgb = CoherenceAnalyzer.level_color(lvl)
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(p.get('event_id', '')))}</td>"
                f"<td><span style='color:rgb({rgb[0]},{rgb[1]},{rgb[2]});font-weight:700'>"
                f"{CoherenceAnalyzer.level_emoji(lvl)} {lvl}</span></td>"
                f"<td>{p.get('risk_score', 0):.1f}</td>"
                f"<td>{p.get('lat', '—')}, {p.get('lon', '—')}</td>"
                f"<td>{p.get('area_km2', '—')}</td>"
                f"<td>{p.get('max_subsidence_mm', '—')}</td>"
                f"<td>{p.get('mean_delta_coh', '—')}</td>"
                "</tr>"
            )
        return (
            "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
            "<thead><tr style='border-bottom:1px solid rgba(255,75,75,0.4);text-align:left'>"
            "<th>Event</th><th>Level</th><th>Score</th><th>Lat, Lon</th>"
            "<th>Area km²</th><th>Max subsidence mm</th><th>Δ coherence</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )


def terrain_3d(dem, layer, bounds, layer_name: str, unit: str, colorscale, cmin, cmax,
               events: dict | None = None, sites: list | None = None, step: int = 3,
               exaggeration: float = 1.6, height: int = 720) -> go.Figure:
    """Real terrain as a 3D surface, draped with a computed layer. Hover = values."""
    lat_min, lat_max, lon_min, lon_max = bounds
    h, w = dem.shape
    z = dem[::step, ::step]
    c = np.asarray(layer)[::step, ::step]
    lats = np.linspace(lat_max, lat_min, h)[::step]
    lons = np.linspace(lon_min, lon_max, w)[::step]
    fig = go.Figure(go.Surface(
        x=lons, y=lats, z=z, surfacecolor=c, colorscale=colorscale, cmin=cmin, cmax=cmax,
        colorbar=dict(title=dict(text=unit, side="right"), thickness=12, len=0.6),
        customdata=np.dstack([c, z]),
        hovertemplate=("%{y:.3f}°N  %{x:.3f}°E<br>Elevation %{customdata[1]:.0f} m<br>"
                       + layer_name + " <b>%{customdata[0]:.2f}</b> " + unit + "<extra></extra>"),
        lighting=dict(ambient=0.55, diffuse=0.8, roughness=0.9, specular=0.05),
        lightposition=dict(x=-1e5, y=1e5, z=4e4),
    ))

    def _elev(la, lo):
        r = int(np.clip((lat_max - la) / (lat_max - lat_min) * (h - 1), 0, h - 1))
        q = int(np.clip((lo - lon_min) / (lon_max - lon_min) * (w - 1), 0, w - 1))
        return float(dem[r, q])

    if events and events.get("features"):
        ps = [f["properties"] for f in events["features"] if "lat" in f["properties"]]
        fig.add_trace(go.Scatter3d(
            x=[p["lon"] for p in ps], y=[p["lat"] for p in ps],
            z=[_elev(p["lat"], p["lon"]) + 250 for p in ps], mode="markers+text",
            text=[p["event_id"] for p in ps], textposition="top center",
            textfont=dict(color="#ffffff", size=11),
            marker=dict(size=6, color=["rgb(%d,%d,%d)" % tuple(CoherenceAnalyzer.level_color(
                p["risk_level"])) for p in ps], line=dict(color="#000", width=1)),
            customdata=[[p["risk_level"], p["risk_score"], p.get("max_subsidence_mm"),
                         p.get("area_km2")] for p in ps],
            hovertemplate=("<b>%{text}</b> %{customdata[0]} (%{customdata[1]})<br>"
                           "Max subsidence %{customdata[2]} mm<br>Area %{customdata[3]} km²"
                           "<extra></extra>"),
            name="Detected events"))
    if sites:
        fig.add_trace(go.Scatter3d(
            x=[s["lng"] for s in sites], y=[s["lat"] for s in sites],
            z=[_elev(s["lat"], s["lng"]) + 400 for s in sites], mode="markers+text",
            text=[s["name"] for s in sites], textposition="top center",
            textfont=dict(color="#e6e6e6", size=11),
            marker=dict(size=4, color="#ffffff", symbol="diamond"),
            customdata=[s["note"] for s in sites],
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra>known site (≈)</extra>",
            name="Known landslide sites"))

    x_km = (lon_max - lon_min) * 111.32 * np.cos(np.radians((lat_min + lat_max) / 2))
    y_km = (lat_max - lat_min) * 111.32
    z_km = (float(dem.max()) - float(dem.min())) / 1000 * exaggeration
    s = max(x_km, y_km)
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), showlegend=True,
        legend=dict(orientation="h", y=0.02, x=0.01, bgcolor="rgba(0,0,0,0.3)"),
        scene=dict(
            aspectmode="manual", aspectratio=dict(x=x_km / s, y=y_km / s, z=z_km / s),
            xaxis=dict(title="Lon", showbackground=False, gridcolor="#333"),
            yaxis=dict(title="Lat", showbackground=False, gridcolor="#333"),
            zaxis=dict(title="m", showbackground=False, gridcolor="#333"),
            camera=dict(eye=dict(x=0.15, y=-1.35, z=0.85)),
        ),
    )
    return fig
