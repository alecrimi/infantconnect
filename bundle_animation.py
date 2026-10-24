"""Bundle-style tract animation seeded at EEG sensors.

Usage in your main script:

    from bundle_animation import BundleAnimationMixin
    class InfantBrainViewer(BundleAnimationMixin, QMainWindow):
        ...

The mixin overrides _setup_animation() and _animate_step(), so the old versions
(and _polyline_between / _point_on_streamline / _move_actor_to) can be deleted
or left in place; they are no longer used.
"""
import numpy as np
import pyvista as pv

N_BUNDLES = 3            # bundles glowing at the same time, each at its own sensor
FIBERS_PER_BUNDLE = 40   # fibers travelling together in one bundle
CANDIDATES = 200         # nearest fibers considered around the sensor
DIRECTION_COHERENCE = 0.6  # min |cos| between a fiber and the lead fiber direction
TRAIL_SAMPLES = 24
FRAME_DT = 0.03
SPEED_FACTOR = 2     # bundle speed in median-fiber-lengths per second  


class BundleAnimationMixin:

    # ------------------------------------------------------------------ setup
    def _animation_electrodes(self):
        """Electrode positions in the same (brain-fitted) frame as the tracts."""
        requested_age = self.age_slider.value()
        age = self._nearest_age(requested_age)
        name = self._pick_scalp_template(requested_age)
        if name is None:
            return {}
        scalp = self.scalp_templates[name]
        electrodes = self.electrode_templates[name]
        parts = [m for m in (self._get_mesh(age, h, "outer") for h in ("left", "right"))
                 if m is not None]
        if not parts:
            return {}
        reference = parts[0] if len(parts) == 1 else parts[0].merge(parts[1:])
        _, electrodes_fit = fit_template_to_brain_ref(scalp, electrodes, reference)
        return electrodes_fit

    def _setup_animation(self, tract_mesh):
        self.anim_timer.stop()
        self.anim_pulses.clear()
        self._anim_rng = np.random.default_rng()

        if tract_mesh.n_cells < 1:
            return

        electrodes = self._animation_electrodes()
        if not electrodes:
            return
        self._anim_electrodes = electrodes

        # Per-fiber point arrays (ordered along the streamline).
        fibers = []
        for cid in range(tract_mesh.n_cells):
            try:
                pts = np.asarray(tract_mesh.get_cell(cid).points, dtype=float)
            except Exception:
                continue
            if len(pts) >= 3:
                fibers.append(pts)
        if not fibers:
            return
        self._anim_fibers = fibers
        self._anim_all_pts = np.vstack(fibers)
        self._anim_cell_of_pt = np.concatenate(
            [np.full(len(p), i, dtype=np.int64) for i, p in enumerate(fibers)]
        )
        self._anim_pt_in_cell = np.concatenate(
            [np.arange(len(p), dtype=np.int64) for p in fibers]
        )

        used = set()
        for b in range(N_BUNDLES):
            bundle = self._make_bundle(b, avoid=used)
            if bundle is None:
                continue
            used.add(bundle["electrode"])
            # Stagger the bundles so they do not all fire in sync.
            bundle["d"] = -bundle["cycle_len"] * (b / max(N_BUNDLES, 1)) * 0.6
            self.anim_pulses.append(bundle)

        if self.anim_pulses:
            self.anim_timer.start(30)

    def _make_bundle(self, slot, avoid=()):
        """Pick a random sensor and gather the coherent fibers passing near it."""
        names = [n for n in self._anim_electrodes if n not in avoid]
        if not names:
            names = list(self._anim_electrodes)
        for _ in range(10):  # retry if a sensor has no usable fibers nearby
            name = names[int(self._anim_rng.integers(len(names)))]
            e = np.asarray(self._anim_electrodes[name], dtype=float)

            dist = np.linalg.norm(self._anim_all_pts - e, axis=1)
            order = np.argsort(dist)

            # Nearest point per fiber, for the CANDIDATES nearest fibers.
            seen, cand = set(), []
            for pi in order:
                ci = int(self._anim_cell_of_pt[pi])
                if ci in seen:
                    continue
                seen.add(ci)
                cand.append((ci, int(self._anim_pt_in_cell[pi])))
                if len(cand) >= CANDIDATES:
                    break

            # Local tangent at the closest point; align everything to the lead fiber.
            tangents = []
            for ci, k in cand:
                pts = self._anim_fibers[ci]
                t = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
                n = np.linalg.norm(t)
                tangents.append(t / n if n > 1e-12 else np.zeros(3))
            lead = next((t for t in tangents if np.linalg.norm(t) > 0), None)
            if lead is None:
                continue

            chosen = []
            for (ci, k), t in zip(cand, tangents):
                c = float(np.dot(t, lead))
                if abs(c) < DIRECTION_COHERENCE:
                    continue
                pts = self._anim_fibers[ci]
                # Orient along the shared direction; start at the sensor-nearest point.
                path = pts[k:] if c > 0 else pts[:k + 1][::-1]
                if len(path) < 3:
                    continue
                seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
                arc = np.concatenate(([0.0], np.cumsum(seg)))
                if arc[-1] <= 1e-9:
                    continue
                chosen.append({"pts": path, "arc": arc, "L": float(arc[-1])})
                if len(chosen) >= FIBERS_PER_BUNDLE:
                    break
            if len(chosen) < 5:
                continue

            lengths = np.array([f["L"] for f in chosen])
            med = float(np.median(lengths))
            trail = 0.20 * med
            for f in chosen:  # small per-fiber lead/lag keeps the bundle organic
                f["offset"] = float(self._anim_rng.uniform(0.0, 0.04 * med))

            tag = f"{slot}"
            ph_line = pv.Line((0, 0, 0), (0, 0, 0.001))
            ph_pts = pv.PolyData(np.zeros((1, 3)))
            halo = self.plotter.add_mesh(
                ph_line, color="#00BFFF", line_width=7, opacity=0.25,
                lighting=False, reset_camera=False, name=f"bundle_halo_{tag}")
            core = self.plotter.add_mesh(
                ph_line.copy(), color="#7DEBFF", line_width=2.0, opacity=0.95,
                lighting=False, reset_camera=False, name=f"bundle_core_{tag}")
            head_halo = self.plotter.add_mesh(
                ph_pts, color="#00CFFF", opacity=0.30, point_size=14,
                render_points_as_spheres=True, lighting=False,
                reset_camera=False, name=f"bundle_head_halo_{tag}")
            head_core = self.plotter.add_mesh(
                ph_pts.copy(), color="#FFF4CF", opacity=1.0, point_size=5,
                render_points_as_spheres=True, lighting=False,
                reset_camera=False, name=f"bundle_head_core_{tag}")

            return {
                "slot": slot, "electrode": name, "fibers": chosen,
                "d": 0.0, "speed": SPEED_FACTOR * med, "trail": trail,
                "cycle_len": float(lengths.max()) + trail,
                "halo": halo, "core": core,
                "head_halo": head_halo, "head_core": head_core,
            }
        return None

    # ---------------------------------------------------------------- frames
    def _animate_step(self):
        if not self.tracts_checkbox.isChecked() or not self.animate_checkbox.isChecked():
            self.anim_timer.stop()
            return

        for bi, bundle in enumerate(self.anim_pulses):
            bundle["d"] += bundle["speed"] * FRAME_DT

            # Finished: re-seed this bundle at a new random sensor.
            if bundle["d"] > bundle["cycle_len"]:
                others = {b["electrode"] for j, b in enumerate(self.anim_pulses) if j != bi}
                new = self._make_bundle(bundle["slot"], avoid=others | {bundle["electrode"]})
                if new is not None:
                    self.anim_pulses[bi] = bundle = new
                else:
                    bundle["d"] = 0.0

            if bundle["d"] < 0:  # staggered start: not launched yet
                self._set_bundle_geometry(bundle, None, None)
                continue

            trail = bundle["trail"]
            all_pts, all_lines, heads = [], [], []
            base = 0
            for f in bundle["fibers"]:
                head = bundle["d"] + f["offset"]
                if head <= 0:
                    continue
                head = min(head, f["L"])
                tail = max(0.0, head - trail)
                if f["L"] - tail < 1e-9 or (head >= f["L"] and tail >= f["L"] - 1e-9):
                    continue
                dists = np.linspace(tail, head, TRAIL_SAMPLES)
                coords = np.column_stack(
                    [np.interp(dists, f["arc"], f["pts"][:, a]) for a in range(3)]
                )
                all_pts.append(coords)
                all_lines.append(np.hstack(
                    ([TRAIL_SAMPLES], base + np.arange(TRAIL_SAMPLES, dtype=np.int64))))
                base += TRAIL_SAMPLES
                if head < f["L"]:
                    heads.append(coords[-1])

            if not all_pts:
                self._set_bundle_geometry(bundle, None, None)
                continue

            trail_poly = pv.PolyData(np.vstack(all_pts), lines=np.concatenate(all_lines))
            head_poly = pv.PolyData(np.array(heads)) if heads else None
            self._set_bundle_geometry(bundle, trail_poly, head_poly)

        self.plotter.render()

    @staticmethod
    def _set_bundle_geometry(bundle, trail_poly, head_poly):
        empty_lines = pv.Line((0, 0, 0), (0, 0, 1e-6))
        empty_pts = pv.PolyData(np.zeros((1, 3)))
        t = trail_poly if trail_poly is not None else empty_lines
        h = head_poly if head_poly is not None else empty_pts
        bundle["halo"].mapper.SetInputData(t)
        bundle["core"].mapper.SetInputData(t.copy())
        bundle["head_halo"].mapper.SetInputData(h)
        bundle["head_core"].mapper.SetInputData(h.copy())


def fit_template_to_brain_ref(scalp, electrodes, reference):
    """Late import so this module can sit next to the main script without a cycle."""
    import importlib
    main_mod = importlib.import_module("__main__")
    return main_mod.fit_template_to_brain(scalp, electrodes, reference)
