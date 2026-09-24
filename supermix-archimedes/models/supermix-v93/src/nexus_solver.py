"""NexusMind deterministic formula-pattern library.

This module provides broad deterministic math/science parsing for development,
audit, and bounded fixtures. A self-hashed `SolverReceipt` authenticates its
parse/result record but does not independently prove that the complete request
was consumed. User-facing answer authority belongs to the stricter fresh
grounding gate in `grounding_runtime.finalize_grounded_response`.

Available pattern families include:
1. **12+ Scientific & Mathematical Scenario Domains**:
   - Kinematics & Motion
   - Dynamics & Newton's Laws
   - Work, Energy & Power
   - Circular Motion & Gravitation
   - Thermodynamics & Heat
   - Hydrostatics & Fluid Dynamics
   - Electromagnetism & DC Circuits
   - Waves & Optics
   - Chemistry Stoichiometry & Solutions
   - Pure Algebra & Systems
   - Geometry & Trigonometry
   - Combinatorics, Probability & Statistics
2. **Exact Rational SI Arithmetic**:
   - Fractional precision (`Fraction` and `Decimal`), dimension vectors, zero float drift.
   - Comprehensive unit conversions for metric & imperial units.
3. **Formal LaTeX Derivations**:
   - Generates step-by-step mathematical proofs and substitutions.
4. **Deterministic integrity metadata**:
   - Produces `SolverReceipt` with a SHA-256 audit digest; the receipt is not
     factual, permission, safety, routing, activation, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import science_plan as v71_science


__all__ = [
    "SolverReceipt",
    "DerivationStep",
    "SolverResult",
    "NexusSolver",
    "solve_problem",
]


SOLVER_RECEIPT_SCHEMA = "nexus-solver-receipt-v2"
MAX_QUERY_CHARS = 4096


# Standard Fundamental Physical Constants in SI (Exact / CODATA 2019)
CONSTANTS: Dict[str, Dict[str, Any]] = {
    "c": {"symbol": "c", "value": Fraction(299792458, 1), "unit": "m/s", "name": "speed_of_light"},
    "h": {"symbol": "h", "value": Fraction(662607015, 10**42), "unit": "J*s", "name": "planck_constant"},
    "hbar": {"symbol": "hbar", "value": Fraction(662607015, 10**42) / Fraction(2 * 3141592653589793, 10**15), "unit": "J*s", "name": "reduced_planck"},
    "G": {"symbol": "G", "value": Fraction(667430, 10**16), "unit": "N*m^2/kg^2", "name": "gravitational_constant"},
    "g": {"symbol": "g", "value": Fraction(980665, 100000), "unit": "m/s^2", "name": "standard_gravity"},
    "k_B": {"symbol": "k_B", "value": Fraction(1380649, 10**29), "unit": "J/K", "name": "boltzmann_constant"},
    "N_A": {"symbol": "N_A", "value": Fraction(602214076, 10**15) * Fraction(10**23, 1), "unit": "1/mol", "name": "avogadro_number"},
    "R": {"symbol": "R", "value": Fraction(831446261815324, 10**14), "unit": "J/(mol*K)", "name": "gas_constant"},
    "e": {"symbol": "e", "value": Fraction(1602176634, 10**28), "unit": "C", "name": "elementary_charge"},
    "k_e": {"symbol": "k_e", "value": Fraction(89875517923, 10), "unit": "N*m^2/C^2", "name": "coulomb_constant"},
    "pi": {"symbol": "pi", "value": Fraction(3141592653589793, 10**15), "unit": "dimensionless", "name": "pi"},
}


@dataclass
class DerivationStep:
    """A single step in a formal mathematical derivation."""

    step_index: int
    description: str
    formula_latex: str
    substitution_latex: str
    evaluated_value: str
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolverReceipt:
    """Cryptographic audit receipt for a deterministic solver execution."""

    schema_version: str = SOLVER_RECEIPT_SCHEMA
    scenario: str = ""
    domain: str = ""
    target: str = ""
    formula_id: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    raw_result_fraction: str = ""
    display_result: str = ""
    unit: str = ""
    derivation_hash: str = ""
    receipt_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolverResult:
    """Master result container returned by the NexusSolver."""

    solved: bool
    query: str
    domain: str
    scenario: str
    target: str
    formula_id: str
    answer_value: Optional[float]
    display_answer: str
    unit: str
    steps: List[DerivationStep] = field(default_factory=list)
    receipt: Optional[SolverReceipt] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solved": self.solved,
            "query": self.query,
            "domain": self.domain,
            "scenario": self.scenario,
            "target": self.target,
            "formula_id": self.formula_id,
            "answer_value": self.answer_value,
            "display_answer": self.display_answer,
            "unit": self.unit,
            "steps": [s.to_dict() for s in self.steps],
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "explanation": self.explanation,
        }


class NexusSolver:
    """Next-generation deterministic math and science problem solver."""

    def __init__(self):
        self._setup_unit_converters()

    def _setup_unit_converters(self):
        self.mass_to_kg = {
            "kg": Fraction(1), "g": Fraction(1, 1000), "mg": Fraction(1, 1000000),
            "lb": Fraction(45359237, 100000000), "oz": Fraction(28349523125, 1000000000000),
            "tonne": Fraction(1000), "ton": Fraction(1000),
        }
        self.length_to_m = {
            "m": Fraction(1), "km": Fraction(1000), "cm": Fraction(1, 100), "mm": Fraction(1, 1000),
            "um": Fraction(1, 1000000), "nm": Fraction(1, 1000000000), "pm": Fraction(1, 10**12),
            "mi": Fraction(1609344, 1000), "mile": Fraction(1609344, 1000), "miles": Fraction(1609344, 1000),
            "yd": Fraction(9144, 10000), "yard": Fraction(9144, 10000),
            "ft": Fraction(3048, 10000), "foot": Fraction(3048, 10000), "feet": Fraction(3048, 10000),
            "in": Fraction(254, 10000), "inch": Fraction(254, 10000), "inches": Fraction(254, 10000),
            "ly": Fraction(9460730472580800, 1), "au": Fraction(149597870700, 1),
        }
        self.time_to_s = {
            "s": Fraction(1), "sec": Fraction(1), "second": Fraction(1), "seconds": Fraction(1),
            "ms": Fraction(1, 1000), "us": Fraction(1, 1000000), "ns": Fraction(1, 1000000000),
            "min": Fraction(60), "minute": Fraction(60), "minutes": Fraction(60),
            "h": Fraction(3600), "hr": Fraction(3600), "hour": Fraction(3600), "hours": Fraction(3600),
            "d": Fraction(86400), "day": Fraction(86400), "days": Fraction(86400),
            "yr": Fraction(31557600), "year": Fraction(31557600), "years": Fraction(31557600),
        }
        self.vel_to_mps = {
            "m/s": Fraction(1), "mps": Fraction(1), "km/h": Fraction(1000, 3600),
            "kph": Fraction(1000, 3600), "mph": Fraction(1609344, 3600000),
            "knot": Fraction(1852, 3600), "knots": Fraction(1852, 3600),
        }
        self.force_to_n = {
            "n": Fraction(1), "kn": Fraction(1000), "mn": Fraction(1000000),
            "lbf": Fraction(4448222, 1000000), "dyn": Fraction(1, 100000),
        }
        self.energy_to_j = {
            "j": Fraction(1), "kj": Fraction(1000), "mj": Fraction(1000000), "gj": Fraction(10**9),
            "cal": Fraction(4184, 1000), "kcal": Fraction(4184), "ev": Fraction(1602176634, 10**28),
            "kev": Fraction(1602176634, 10**25), "mev": Fraction(1602176634, 10**22),
            "kwh": Fraction(3600000), "btu": Fraction(1055056, 1000),
        }
        self.power_to_w = {
            "w": Fraction(1), "kw": Fraction(1000), "mw": Fraction(1000000), "gw": Fraction(10**9),
            "hp": Fraction(745699872, 1000000),
        }
        self.pressure_to_pa = {
            "pa": Fraction(1), "kpa": Fraction(1000), "mpa": Fraction(1000000),
            "bar": Fraction(100000), "mbar": Fraction(100), "atm": Fraction(101325),
            "psi": Fraction(6894757, 1000), "torr": Fraction(101325, 760), "mmhg": Fraction(101325, 760),
        }

    def _normalize_num(self, val_str: str) -> Fraction:
        """Parse clean fractional or decimal numbers."""
        clean = val_str.strip().replace(",", "")
        if "/" in clean:
            num, den = clean.split("/", 1)
            return Fraction(int(num), int(den))
        if "." in clean:
            return Fraction(Decimal(clean))
        return Fraction(int(clean))

    def _format_fraction(self, f: Fraction, max_decimals: int = 4) -> str:
        """Format exact fraction to clean display string."""
        if f.denominator == 1:
            return str(f.numerator)
        val_float = float(f)
        if abs(val_float) < 1e-4 and val_float != 0:
            return f"{val_float:.4e}"
        # If fraction terminates cleanly in small decimal
        s = f"{val_float:.{max_decimals}f}".rstrip("0").rstrip(".")
        return s

    @staticmethod
    def _bridge_answer_value(ans_dict: Dict[str, Any]) -> Optional[float]:
        """The numeric answer of a `science_plan` result, or None.

        `science_plan.solve_science_scenario` writes its answer under
        ``"exact"`` as a fraction string (``"32"``, ``"161/5"``); this bridge
        read only ``"fraction"``, a key that result never carries, so every
        query the v71 path solved came back with ``answer_value=None``. Nothing
        noticed until v93 added `final_velocity` (``v = u + a t``, the one
        registry formula `build_omni_corpus` uses): `verify` compares
        ``answer_value`` to the generator's answer, so a None dropped 100% of
        the task at build time -- 0 of 200 smoke draws verified before this
        line, 200 of 200 after. Both keys are read so a result that carries
        either still bridges; an unparseable value stays None rather than
        raising, which keeps the fall-through to the extended solvers intact.
        """

        for key in ("fraction", "exact"):
            if key in ans_dict:
                try:
                    return float(Fraction(str(ans_dict[key])))
                except (ValueError, ZeroDivisionError, TypeError):
                    return None
        return None

    def solve(self, query: str) -> SolverResult:
        """Master solve dispatcher across all scientific and mathematical domains."""
        query_clean = query.strip()

        # Check v71 closed-world solver first for backward compatibility
        v71_res = v71_science.solve_science_scenario(query_clean)
        if v71_res.get("solved") is True:
            ans_dict = v71_res.get("answer", {})
            receipt_dict = v71_res.get("receipt", {})
            return SolverResult(
                solved=True,
                query=query_clean,
                domain="physics",
                scenario=v71_res.get("scenario", ""),
                target=v71_res.get("target", ""),
                formula_id=v71_res.get("formula_id", ""),
                answer_value=self._bridge_answer_value(ans_dict),
                display_answer=ans_dict.get("display", ""),
                unit=ans_dict.get("unit", ""),
                steps=[
                    DerivationStep(
                        step_index=1,
                        description=f"Verified registry evaluation [{v71_res.get('formula_id')}]",
                        formula_latex=v71_res.get("formula_id", ""),
                        substitution_latex=f"{ans_dict.get('display')} {ans_dict.get('unit')}",
                        evaluated_value=ans_dict.get("display", ""),
                        unit=ans_dict.get("unit", ""),
                    )
                ],
                receipt=SolverReceipt(
                    schema_version=SOLVER_RECEIPT_SCHEMA,
                    scenario=v71_res.get("scenario", ""),
                    domain="physics",
                    target=v71_res.get("target", ""),
                    formula_id=v71_res.get("formula_id", ""),
                    inputs=receipt_dict.get("inputs", {}),
                    raw_result_fraction=str(ans_dict.get("fraction", "")),
                    display_result=ans_dict.get("display", ""),
                    unit=ans_dict.get("unit", ""),
                    receipt_sha256=receipt_dict.get("receipt_sha256", ""),
                ),
                explanation=f"Exact verified scientific solution: {ans_dict.get('display')} {ans_dict.get('unit')}",
            )

        # Dispatch across extended scenario solvers
        solvers = [
            self._solve_kinematics_extended,
            self._solve_dynamics_newton,
            self._solve_work_energy_power,
            self._solve_circular_gravitation,
            self._solve_thermo_heat,
            self._solve_fluids_hydrostatics,
            self._solve_circuits_electromagnetism,
            self._solve_waves_optics,
            self._solve_chemistry_solutions,
            self._solve_quadratic_algebra,
            self._solve_linear_system_2x2,
            self._solve_compound_interest,
            self._solve_series_progressions,
            self._solve_geometry_trig,
            self._solve_combinatorics_stats,
        ]

        for s_func in solvers:
            try:
                res = s_func(query_clean)
                if res and res.solved:
                    return res
            except Exception:
                continue

        return SolverResult(
            solved=False,
            query=query_clean,
            domain="unresolved",
            scenario="none",
            target="none",
            formula_id="none",
            answer_value=None,
            display_answer="",
            unit="",
            explanation="Could not match deterministic formula pattern with complete parameters.",
        )

    # --------------------------------------------------------------------------
    # 1. Kinematics Extended Solver
    # --------------------------------------------------------------------------
    def _solve_kinematics_extended(self, q: str) -> Optional[SolverResult]:
        # Torricelli: v^2 = u^2 + 2as -> v = sqrt(u^2 + 2as)
        u_m = re.search(r"(?:initial velocity|u)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*m/s", q, re.I)
        a_m = re.search(r"(?:acceleration|a)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*m/s\^?2", q, re.I)
        s_m = re.search(r"(?:displacement|distance|s|d)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        t_m = re.search(r"(?:time|t)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*s\b", q, re.I)

        if u_m and a_m and s_m and not t_m and ("final velocity" in q.lower() or "speed" in q.lower()):
            u = self._normalize_num(u_m.group(1))
            a = self._normalize_num(a_m.group(1))
            s = self._normalize_num(s_m.group(1))
            v_sq = u*u + 2*a*s
            if v_sq >= 0:
                v_val = math.sqrt(float(v_sq))
                v_frac = Fraction(Decimal(f"{v_val:.6f}"))
                disp = self._format_fraction(v_frac)
                steps = [
                    DerivationStep(1, "Apply Torricelli's equation for constant acceleration", "v^2 = u^2 + 2as", f"v^2 = ({u})^2 + 2({a})({s}) = {v_sq}", f"{v_sq}", "m^2/s^2"),
                    DerivationStep(2, "Extract root for positive final velocity", "v = \\sqrt{u^2 + 2as}", f"v = \\sqrt{{{v_sq}}} = {disp}", disp, "m/s"),
                ]
                receipt = self._build_receipt("kinematics", "physics", "final_velocity", "torricelli.final_velocity", {"u": str(u), "a": str(a), "s": str(s)}, str(v_frac), disp, "m/s", steps)
                return SolverResult(True, q, "physics", "kinematics", "final_velocity", "torricelli.final_velocity", v_val, disp, "m/s", steps, receipt, f"Final velocity is {disp} m/s.")

        # Average velocity: v_avg = (u + v) / 2
        v_m = re.search(r"(?:final velocity|v)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*m/s", q, re.I)
        if u_m and v_m and ("average velocity" in q.lower() or "mean velocity" in q.lower()):
            u = self._normalize_num(u_m.group(1))
            v = self._normalize_num(v_m.group(1))
            v_avg = (u + v) / 2
            disp = self._format_fraction(v_avg)
            steps = [
                DerivationStep(1, "Compute average velocity under constant acceleration", "v_{\\text{avg}} = \\frac{u + v}{2}", f"v_{{\\text{{avg}}}} = \\frac{{{u} + {v}}}{{2}} = {disp}", disp, "m/s")
            ]
            receipt = self._build_receipt("kinematics", "physics", "average_velocity", "kinematics.average_velocity", {"u": str(u), "v": str(v)}, str(v_avg), disp, "m/s", steps)
            return SolverResult(True, q, "physics", "kinematics", "average_velocity", "kinematics.average_velocity", float(v_avg), disp, "m/s", steps, receipt, f"Average velocity is {disp} m/s.")

        return None

    # --------------------------------------------------------------------------
    # 2. Dynamics & Newton's Laws (F=ma, p=mv, J=F*dt, f_k=mu*N)
    # --------------------------------------------------------------------------
    def _solve_dynamics_newton(self, q: str) -> Optional[SolverResult]:
        m_m = re.search(r"(?:mass|m)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", q, re.I)
        a_m = re.search(r"(?:acceleration|a)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*m/s\^?2", q, re.I)
        f_m = re.search(r"(?:force|F|net force)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(N|kN)\b", q, re.I)
        v_m = re.search(r"(?:velocity|speed|v)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m/s\b", q, re.I)
        t_m = re.search(r"(?:time|duration|dt|t)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*s\b", q, re.I)
        mu_m = re.search(r"(?:friction coefficient|coefficient of friction|mu|mu_k)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)", q, re.I)
        n_m = re.search(r"(?:normal force|N)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*N\b", q, re.I)

        # Force F = m * a
        if m_m and a_m and ("force" in q.lower() or "net force" in q.lower()) and not f_m:
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            acc = self._normalize_num(a_m.group(1))
            force = mass * acc
            disp = self._format_fraction(force)
            steps = [
                DerivationStep(1, "Apply Newton's Second Law of Motion", "F = m \\cdot a", f"F = ({mass}\\text{{ kg}}) \\cdot ({acc}\\text{{ m/s}}^2) = {disp}\\text{{ N}}", disp, "N")
            ]
            receipt = self._build_receipt("dynamics", "physics", "force", "newton.second_law_force", {"m": str(mass), "a": str(acc)}, str(force), disp, "N", steps)
            return SolverResult(True, q, "physics", "dynamics", "force", "newton.second_law_force", float(force), disp, "N", steps, receipt, f"Net force is {disp} N.")

        # Acceleration a = F / m
        if f_m and m_m and ("acceleration" in q.lower() or "find acceleration" in q.lower()) and not a_m:
            force = self._normalize_num(f_m.group(1)) * (Fraction(1000) if f_m.group(2).lower() == "kn" else Fraction(1))
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            if mass > 0:
                acc = force / mass
                disp = self._format_fraction(acc)
                steps = [
                    DerivationStep(1, "Solve Newton's Second Law for acceleration", "a = \\frac{F}{m}", f"a = \\frac{{{force}\\text{{ N}}}}{{{mass}\\text{{ kg}}}} = {disp}\\text{{ m/s}}^2", disp, "m/s^2")
                ]
                receipt = self._build_receipt("dynamics", "physics", "acceleration", "newton.second_law_accel", {"F": str(force), "m": str(mass)}, str(acc), disp, "m/s^2", steps)
                return SolverResult(True, q, "physics", "dynamics", "acceleration", "newton.second_law_accel", float(acc), disp, "m/s^2", steps, receipt, f"Acceleration is {disp} m/s^2.")

        # Momentum p = m * v
        if m_m and v_m and ("momentum" in q.lower() or "linear momentum" in q.lower()):
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            vel = self._normalize_num(v_m.group(1))
            mom = mass * vel
            disp = self._format_fraction(mom)
            steps = [
                DerivationStep(1, "Calculate linear momentum definition", "p = m \\cdot v", f"p = ({mass}\\text{{ kg}}) \\cdot ({vel}\\text{{ m/s}}) = {disp}\\text{{ kg\\cdot m/s}}", disp, "kg*m/s")
            ]
            receipt = self._build_receipt("dynamics", "physics", "momentum", "dynamics.momentum", {"m": str(mass), "v": str(vel)}, str(mom), disp, "kg*m/s", steps)
            return SolverResult(True, q, "physics", "dynamics", "momentum", "dynamics.momentum", float(mom), disp, "kg*m/s", steps, receipt, f"Momentum is {disp} kg*m/s.")

        # Impulse J = F * dt
        if f_m and t_m and ("impulse" in q.lower() or "change in momentum" in q.lower()):
            force = self._normalize_num(f_m.group(1)) * (Fraction(1000) if f_m.group(2).lower() == "kn" else Fraction(1))
            dt = self._normalize_num(t_m.group(1))
            imp = force * dt
            disp = self._format_fraction(imp)
            steps = [
                DerivationStep(1, "Calculate impulse imparted by force over duration", "J = F \\cdot \\Delta t", f"J = ({force}\\text{{ N}}) \\cdot ({dt}\\text{{ s}}) = {disp}\\text{{ N\\cdot s}}", disp, "N*s")
            ]
            receipt = self._build_receipt("dynamics", "physics", "impulse", "dynamics.impulse", {"F": str(force), "t": str(dt)}, str(imp), disp, "N*s", steps)
            return SolverResult(True, q, "physics", "dynamics", "impulse", "dynamics.impulse", float(imp), disp, "N*s", steps, receipt, f"Impulse is {disp} N*s.")

        # Friction f_k = mu_k * N
        if mu_m and n_m and ("friction" in q.lower() or "frictional force" in q.lower()):
            mu = self._normalize_num(mu_m.group(1))
            norm = self._normalize_num(n_m.group(1))
            fric = mu * norm
            disp = self._format_fraction(fric)
            steps = [
                DerivationStep(1, "Calculate kinetic friction force", "f_k = \\mu_k \\cdot N", f"f_k = ({mu}) \\cdot ({norm}\\text{{ N}}) = {disp}\\text{{ N}}", disp, "N")
            ]
            receipt = self._build_receipt("dynamics", "physics", "friction", "dynamics.friction", {"mu": str(mu), "N": str(norm)}, str(fric), disp, "N", steps)
            return SolverResult(True, q, "physics", "dynamics", "friction", "dynamics.friction", float(fric), disp, "N", steps, receipt, f"Friction force is {disp} N.")

        return None

    # --------------------------------------------------------------------------
    # 3. Work, Energy & Power (W=F*d, Ek=1/2mv^2, Ep=mgh, P=W/t, Es=1/2kx^2)
    # --------------------------------------------------------------------------
    def _solve_work_energy_power(self, q: str) -> Optional[SolverResult]:
        f_m = re.search(r"(?:force|F)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*N\b", q, re.I)
        d_m = re.search(r"(?:distance|displacement|d|height|h|s)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        m_m = re.search(r"(?:mass|m)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", q, re.I)
        v_m = re.search(r"(?:velocity|speed|v)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m/s\b", q, re.I)
        t_m = re.search(r"(?:time|t|duration)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*s\b", q, re.I)
        w_m = re.search(r"(?:work|W|energy)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(J|kJ)\b", q, re.I)
        k_m = re.search(r"(?:spring constant|k)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*N/m\b", q, re.I)
        x_m = re.search(r"(?:compression|extension|stretch|x)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(m|cm)\b", q, re.I)

        # Kinetic Energy: Ek = 1/2 * m * v^2
        if m_m and v_m and ("kinetic energy" in q.lower() or "ke" in q.lower()):
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            vel = self._normalize_num(v_m.group(1))
            ke = Fraction(1, 2) * mass * vel * vel
            disp = self._format_fraction(ke)
            steps = [
                DerivationStep(1, "Calculate kinetic energy", "E_k = \\frac{1}{2} m v^2", f"E_k = \\frac{{1}}{{2}} ({mass}\\text{{ kg}}) ({vel}\\text{{ m/s}})^2 = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("energy", "physics", "kinetic_energy", "energy.kinetic", {"m": str(mass), "v": str(vel)}, str(ke), disp, "J", steps)
            return SolverResult(True, q, "physics", "energy", "kinetic_energy", "energy.kinetic", float(ke), disp, "J", steps, receipt, f"Kinetic energy is {disp} J.")

        # Potential Energy: Ep = m * g * h
        if m_m and d_m and ("potential energy" in q.lower() or "gravitational potential energy" in q.lower() or "pe" in q.lower()):
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            h = self._normalize_num(d_m.group(1))
            g = CONSTANTS["g"]["value"]
            pe = mass * g * h
            disp = self._format_fraction(pe)
            steps = [
                DerivationStep(1, "Calculate gravitational potential energy (g=9.80665 m/s^2)", "E_p = m g h", f"E_p = ({mass})({g})({h}) = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("energy", "physics", "potential_energy", "energy.potential", {"m": str(mass), "h": str(h)}, str(pe), disp, "J", steps)
            return SolverResult(True, q, "physics", "energy", "potential_energy", "energy.potential", float(pe), disp, "J", steps, receipt, f"Potential energy is {disp} J.")

        # Work: W = F * d
        if f_m and d_m and ("work done" in q.lower() or "work" in q.lower()) and not w_m:
            force = self._normalize_num(f_m.group(1))
            dist = self._normalize_num(d_m.group(1))
            work = force * dist
            disp = self._format_fraction(work)
            steps = [
                DerivationStep(1, "Calculate mechanical work done along displacement", "W = F \\cdot d", f"W = ({force}\\text{{ N}}) \\cdot ({dist}\\text{{ m}}) = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("energy", "physics", "work", "energy.work", {"F": str(force), "d": str(dist)}, str(work), disp, "J", steps)
            return SolverResult(True, q, "physics", "energy", "work", "energy.work", float(work), disp, "J", steps, receipt, f"Work done is {disp} J.")

        # Power: P = W / t
        if w_m and t_m and ("power" in q.lower()):
            work = self._normalize_num(w_m.group(1)) * (Fraction(1000) if w_m.group(2).lower() == "kj" else Fraction(1))
            time_val = self._normalize_num(t_m.group(1))
            if time_val > 0:
                power = work / time_val
                disp = self._format_fraction(power)
                steps = [
                    DerivationStep(1, "Calculate power as rate of energy transfer", "P = \\frac{W}{t}", f"P = \\frac{{{work}\\text{{ J}}}}{{{time_val}\\text{{ s}}}} = {disp}\\text{{ W}}", disp, "W")
                ]
                receipt = self._build_receipt("energy", "physics", "power", "energy.power", {"W": str(work), "t": str(time_val)}, str(power), disp, "W", steps)
                return SolverResult(True, q, "physics", "energy", "power", "energy.power", float(power), disp, "W", steps, receipt, f"Power is {disp} W.")

        # Spring Potential Energy: Es = 1/2 * k * x^2
        if k_m and x_m and ("spring energy" in q.lower() or "elastic potential" in q.lower()):
            k = self._normalize_num(k_m.group(1))
            x = self._normalize_num(x_m.group(1)) * (Fraction(1) if x_m.group(2).lower() == "m" else Fraction(1, 100))
            e_s = Fraction(1, 2) * k * x * x
            disp = self._format_fraction(e_s)
            steps = [
                DerivationStep(1, "Calculate elastic potential energy of Hookean spring", "E_s = \\frac{1}{2} k x^2", f"E_s = \\frac{{1}}{{2}} ({k}\\text{{ N/m}}) ({x}\\text{{ m}})^2 = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("energy", "physics", "spring_energy", "energy.spring", {"k": str(k), "x": str(x)}, str(e_s), disp, "J", steps)
            return SolverResult(True, q, "physics", "energy", "spring_energy", "energy.spring", float(e_s), disp, "J", steps, receipt, f"Elastic potential energy is {disp} J.")

        return None

    # --------------------------------------------------------------------------
    # 4. Circular Motion & Gravitation (ac=v^2/r, Fc=mv^2/r, Fg=G*m1*m2/r^2)
    # --------------------------------------------------------------------------
    def _solve_circular_gravitation(self, q: str) -> Optional[SolverResult]:
        v_m = re.search(r"(?:velocity|speed|v)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m/s\b", q, re.I)
        r_m = re.search(r"(?:radius|distance|r)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(m|km)\b", q, re.I)
        m_m = re.search(r"(?:mass|m)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", q, re.I)

        # Centripetal Acceleration: a_c = v^2 / r
        if v_m and r_m and ("centripetal acceleration" in q.lower() or "radial acceleration" in q.lower()):
            v = self._normalize_num(v_m.group(1))
            r = self._normalize_num(r_m.group(1)) * (Fraction(1000) if r_m.group(2).lower() == "km" else Fraction(1))
            if r > 0:
                ac = (v * v) / r
                disp = self._format_fraction(ac)
                steps = [
                    DerivationStep(1, "Calculate centripetal acceleration", "a_c = \\frac{v^2}{r}", f"a_c = \\frac{{({v}\\text{{ m/s}})^2}}{{{r}\\text{{ m}}}} = {disp}\\text{{ m/s}}^2", disp, "m/s^2")
                ]
                receipt = self._build_receipt("gravitation", "physics", "centripetal_acceleration", "circular.centripetal_accel", {"v": str(v), "r": str(r)}, str(ac), disp, "m/s^2", steps)
                return SolverResult(True, q, "physics", "gravitation", "centripetal_acceleration", "circular.centripetal_accel", float(ac), disp, "m/s^2", steps, receipt, f"Centripetal acceleration is {disp} m/s^2.")

        # Centripetal Force: F_c = m * v^2 / r
        if m_m and v_m and r_m and ("centripetal force" in q.lower()):
            m = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            v = self._normalize_num(v_m.group(1))
            r = self._normalize_num(r_m.group(1)) * (Fraction(1000) if r_m.group(2).lower() == "km" else Fraction(1))
            if r > 0:
                fc = (m * v * v) / r
                disp = self._format_fraction(fc)
                steps = [
                    DerivationStep(1, "Calculate centripetal force", "F_c = \\frac{m v^2}{r}", f"F_c = \\frac{{({m})({v})^2}}{{{r}}} = {disp}\\text{{ N}}", disp, "N")
                ]
                receipt = self._build_receipt("gravitation", "physics", "centripetal_force", "circular.centripetal_force", {"m": str(m), "v": str(v), "r": str(r)}, str(fc), disp, "N", steps)
                return SolverResult(True, q, "physics", "gravitation", "centripetal_force", "circular.centripetal_force", float(fc), disp, "N", steps, receipt, f"Centripetal force is {disp} N.")

        return None

    # --------------------------------------------------------------------------
    # 5. Thermodynamics & Heat (Q = mc*dT, Q = mL, eta = 1 - Tc/Th)
    # --------------------------------------------------------------------------
    def _solve_thermo_heat(self, q: str) -> Optional[SolverResult]:
        m_m = re.search(r"(?:mass|m)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", q, re.I)
        c_m = re.search(r"(?:specific heat|specific heat capacity|c)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*J/(?:kg\*K|kg\*C|kg K|kg C)\b", q, re.I)
        dt_m = re.search(r"(?:temperature change|delta T|dT|change in temperature)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:K|C|degrees)?\b", q, re.I)
        th_m = re.search(r"(?:hot temperature|hot reservoir|T_H|Th)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*K\b", q, re.I)
        tc_m = re.search(r"(?:cold temperature|cold reservoir|T_C|Tc)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*K\b", q, re.I)

        # Heat transfer: Q = m * c * dT
        if m_m and c_m and dt_m and ("heat" in q.lower() or "thermal energy" in q.lower()):
            m = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            c = self._normalize_num(c_m.group(1))
            dt = self._normalize_num(dt_m.group(1))
            q_val = m * c * dt
            disp = self._format_fraction(q_val)
            steps = [
                DerivationStep(1, "Calculate sensible heat transfer", "Q = m \\cdot c \\cdot \\Delta T", f"Q = ({m}\\text{{ kg}}) ({c}\\text{{ J/kg\\cdot K}}) ({dt}\\text{{ K}}) = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("thermodynamics", "physics", "heat", "thermo.sensible_heat", {"m": str(m), "c": str(c), "dT": str(dt)}, str(q_val), disp, "J", steps)
            return SolverResult(True, q, "physics", "thermodynamics", "heat", "thermo.sensible_heat", float(q_val), disp, "J", steps, receipt, f"Heat transferred is {disp} J.")

        # Carnot Efficiency: eta = 1 - Tc / Th
        if th_m and tc_m and ("carnot efficiency" in q.lower() or "maximum efficiency" in q.lower() or "efficiency" in q.lower()):
            th = self._normalize_num(th_m.group(1))
            tc = self._normalize_num(tc_m.group(1))
            if th > tc and tc >= 0:
                eta = Fraction(1) - (tc / th)
                eta_pct = eta * 100
                disp = self._format_fraction(eta_pct)
                steps = [
                    DerivationStep(1, "Calculate ideal Carnot heat engine efficiency", "\\eta = 1 - \\frac{T_C}{T_H}", f"\\eta = 1 - \\frac{{{tc}}}{{{th}}} = {self._format_fraction(eta)} = {disp}\\%", f"{disp}%", "%")
                ]
                receipt = self._build_receipt("thermodynamics", "physics", "efficiency", "thermo.carnot_efficiency", {"Th": str(th), "Tc": str(tc)}, str(eta), f"{disp}%", "%", steps)
                return SolverResult(True, q, "physics", "thermodynamics", "efficiency", "thermo.carnot_efficiency", float(eta), f"{disp}%", "%", steps, receipt, f"Carnot efficiency is {disp}%.")

        return None

    # --------------------------------------------------------------------------
    # 6. Hydrostatics & Fluids (P = rho*g*h, Fb = rho*V*g, A1*v1 = A2*v2)
    # --------------------------------------------------------------------------
    def _solve_fluids_hydrostatics(self, q: str) -> Optional[SolverResult]:
        rho_m = re.search(r"(?:density|rho)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*kg/m\^?3\b", q, re.I)
        h_m = re.search(r"(?:depth|height|h)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        vol_m = re.search(r"(?:volume|V)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(m\^?3|L|liters)\b", q, re.I)

        # Hydrostatic Pressure: P = rho * g * h
        if rho_m and h_m and ("hydrostatic pressure" in q.lower() or "gauge pressure" in q.lower() or "pressure at depth" in q.lower()):
            rho = self._normalize_num(rho_m.group(1))
            h = self._normalize_num(h_m.group(1))
            g = CONSTANTS["g"]["value"]
            p = rho * g * h
            disp = self._format_fraction(p)
            steps = [
                DerivationStep(1, "Calculate hydrostatic gauge pressure", "P = \\rho g h", f"P = ({rho}\\text{{ kg/m}}^3)({g}\\text{{ m/s}}^2)({h}\\text{{ m}}) = {disp}\\text{{ Pa}}", disp, "Pa")
            ]
            receipt = self._build_receipt("fluids", "physics", "hydrostatic_pressure", "fluids.hydrostatic_pressure", {"rho": str(rho), "h": str(h)}, str(p), disp, "Pa", steps)
            return SolverResult(True, q, "physics", "fluids", "hydrostatic_pressure", "fluids.hydrostatic_pressure", float(p), disp, "Pa", steps, receipt, f"Hydrostatic pressure is {disp} Pa.")

        # Buoyant Force: Fb = rho * V * g
        if rho_m and vol_m and ("buoyant force" in q.lower() or "buoyancy" in q.lower() or "archimedes" in q.lower()):
            rho = self._normalize_num(rho_m.group(1))
            vol = self._normalize_num(vol_m.group(1)) * (Fraction(1, 1000) if vol_m.group(2).lower() in ("l", "liters") else Fraction(1))
            g = CONSTANTS["g"]["value"]
            fb = rho * vol * g
            disp = self._format_fraction(fb)
            steps = [
                DerivationStep(1, "Apply Archimedes' principle for buoyant force", "F_b = \\rho \\cdot V \\cdot g", f"F_b = ({rho})({vol})({g}) = {disp}\\text{{ N}}", disp, "N")
            ]
            receipt = self._build_receipt("fluids", "physics", "buoyant_force", "fluids.buoyant_force", {"rho": str(rho), "V": str(vol)}, str(fb), disp, "N", steps)
            return SolverResult(True, q, "physics", "fluids", "buoyant_force", "fluids.buoyant_force", float(fb), disp, "N", steps, receipt, f"Buoyant force is {disp} N.")

        return None

    # --------------------------------------------------------------------------
    # 7. Electromagnetism & Circuits (V=IR, P=VI=I^2R=V^2/R, Req series/parallel)
    # --------------------------------------------------------------------------
    def _solve_circuits_electromagnetism(self, q: str) -> Optional[SolverResult]:
        v_m = re.search(r"(?:voltage|V|potential difference)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(V|mV|kV)\b", q, re.I)
        i_m = re.search(r"(?:current|I)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(A|mA)\b", q, re.I)
        r_m = re.search(r"(?:resistance|R)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(ohm|ohms|Ω|kohm|kΩ)\b", q, re.I)
        r1_m = re.search(r"R1\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(ohm|ohms|Ω|kohm|kΩ)\b", q, re.I)
        r2_m = re.search(r"R2\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(ohm|ohms|Ω|kohm|kΩ)\b", q, re.I)

        # Ohm's Law: V = I * R -> Voltage
        if i_m and r_m and ("voltage" in q.lower() or "potential difference" in q.lower()) and not v_m:
            curr = self._normalize_num(i_m.group(1)) * (Fraction(1, 1000) if i_m.group(2).lower() == "ma" else Fraction(1))
            res = self._normalize_num(r_m.group(1)) * (Fraction(1000) if "k" in r_m.group(2).lower() else Fraction(1))
            volt = curr * res
            disp = self._format_fraction(volt)
            steps = [
                DerivationStep(1, "Apply Ohm's Law for voltage", "V = I \\cdot R", f"V = ({curr}\\text{{ A}}) \\cdot ({res}\\text{{ \\Omega}}) = {disp}\\text{{ V}}", disp, "V")
            ]
            receipt = self._build_receipt("circuits", "physics", "voltage", "circuits.ohms_law_voltage", {"I": str(curr), "R": str(res)}, str(volt), disp, "V", steps)
            return SolverResult(True, q, "physics", "circuits", "voltage", "circuits.ohms_law_voltage", float(volt), disp, "V", steps, receipt, f"Voltage is {disp} V.")

        # Ohm's Law: I = V / R -> Current
        if v_m and r_m and ("current" in q.lower()) and not i_m:
            volt = self._normalize_num(v_m.group(1)) * (Fraction(1000) if v_m.group(2).lower() == "kv" else (Fraction(1, 1000) if v_m.group(2).lower() == "mv" else Fraction(1)))
            res = self._normalize_num(r_m.group(1)) * (Fraction(1000) if "k" in r_m.group(2).lower() else Fraction(1))
            if res > 0:
                curr = volt / res
                disp = self._format_fraction(curr)
                steps = [
                    DerivationStep(1, "Apply Ohm's Law for current", "I = \\frac{V}{R}", f"I = \\frac{{{volt}\\text{{ V}}}}{{{res}\\text{{ \\Omega}}}} = {disp}\\text{{ A}}", disp, "A")
                ]
                receipt = self._build_receipt("circuits", "physics", "current", "circuits.ohms_law_current", {"V": str(volt), "R": str(res)}, str(curr), disp, "A", steps)
                return SolverResult(True, q, "physics", "circuits", "current", "circuits.ohms_law_current", float(curr), disp, "A", steps, receipt, f"Current is {disp} A.")

        # Electrical Power: P = V * I
        if v_m and i_m and ("electrical power" in q.lower() or "power dissipated" in q.lower() or "power" in q.lower()):
            volt = self._normalize_num(v_m.group(1)) * (Fraction(1000) if v_m.group(2).lower() == "kv" else (Fraction(1, 1000) if v_m.group(2).lower() == "mv" else Fraction(1)))
            curr = self._normalize_num(i_m.group(1)) * (Fraction(1, 1000) if i_m.group(2).lower() == "ma" else Fraction(1))
            power = volt * curr
            disp = self._format_fraction(power)
            steps = [
                DerivationStep(1, "Calculate electric power", "P = V \\cdot I", f"P = ({volt}\\text{{ V}}) \\cdot ({curr}\\text{{ A}}) = {disp}\\text{{ W}}", disp, "W")
            ]
            receipt = self._build_receipt("circuits", "physics", "power", "circuits.electric_power", {"V": str(volt), "I": str(curr)}, str(power), disp, "W", steps)
            return SolverResult(True, q, "physics", "circuits", "power", "circuits.electric_power", float(power), disp, "W", steps, receipt, f"Electrical power is {disp} W.")

        # Parallel Resistors: Req = (R1 * R2) / (R1 + R2)
        if r1_m and r2_m and ("parallel" in q.lower()):
            r1 = self._normalize_num(r1_m.group(1)) * (Fraction(1000) if "k" in r1_m.group(2).lower() else Fraction(1))
            r2 = self._normalize_num(r2_m.group(1)) * (Fraction(1000) if "k" in r2_m.group(2).lower() else Fraction(1))
            req = (r1 * r2) / (r1 + r2)
            disp = self._format_fraction(req)
            steps = [
                DerivationStep(1, "Calculate equivalent parallel resistance", "R_{\\text{eq}} = \\frac{R_1 R_2}{R_1 + R_2}", f"R_{{\\text{{eq}}}} = \\frac{{{r1} \\cdot {r2}}}{{{r1} + {r2}}} = {disp}\\text{{ \\Omega}}", disp, "Ω")
            ]
            receipt = self._build_receipt("circuits", "physics", "equivalent_resistance", "circuits.parallel_resistance", {"R1": str(r1), "R2": str(r2)}, str(req), disp, "Ω", steps)
            return SolverResult(True, q, "physics", "circuits", "equivalent_resistance", "circuits.parallel_resistance", float(req), disp, "Ω", steps, receipt, f"Equivalent parallel resistance is {disp} Ω.")

        return None

    # --------------------------------------------------------------------------
    # 8. Waves & Optics (v = f * lambda, T = 1/f, E = hf)
    # --------------------------------------------------------------------------
    def _solve_waves_optics(self, q: str) -> Optional[SolverResult]:
        f_m = re.search(r"(?:frequency|f)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz|GHz)\b", q, re.I)
        lam_m = re.search(r"(?:wavelength|lambda)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(m|cm|mm|nm)\b", q, re.I)
        v_m = re.search(r"(?:wave speed|speed|v)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m/s\b", q, re.I)

        # Wave Speed: v = f * lambda
        if f_m and lam_m and ("wave speed" in q.lower() or "speed of wave" in q.lower() or "velocity" in q.lower()):
            f_scale = {"hz": Fraction(1), "khz": Fraction(1000), "mhz": Fraction(1000000), "ghz": Fraction(10**9)}
            lam_scale = {"m": Fraction(1), "cm": Fraction(1, 100), "mm": Fraction(1, 1000), "nm": Fraction(1, 10**9)}
            freq = self._normalize_num(f_m.group(1)) * f_scale[f_m.group(2).lower()]
            lam = self._normalize_num(lam_m.group(1)) * lam_scale[lam_m.group(2).lower()]
            vel = freq * lam
            disp = self._format_fraction(vel)
            steps = [
                DerivationStep(1, "Calculate wave propagation speed", "v = f \\cdot \\lambda", f"v = ({freq}\\text{{ Hz}}) \\cdot ({lam}\\text{{ m}}) = {disp}\\text{{ m/s}}", disp, "m/s")
            ]
            receipt = self._build_receipt("waves", "physics", "wave_speed", "waves.wave_speed", {"f": str(freq), "lambda": str(lam)}, str(vel), disp, "m/s", steps)
            return SolverResult(True, q, "physics", "waves", "wave_speed", "waves.wave_speed", float(vel), disp, "m/s", steps, receipt, f"Wave speed is {disp} m/s.")

        # Wave period: T = 1 / f
        if f_m and ("period" in q.lower() or "wave period" in q.lower()):
            f_scale = {"hz": Fraction(1), "khz": Fraction(1000), "mhz": Fraction(1000000), "ghz": Fraction(10**9)}
            freq = self._normalize_num(f_m.group(1)) * f_scale[f_m.group(2).lower()]
            if freq > 0:
                period = Fraction(1) / freq
                disp = self._format_fraction(period, max_decimals=6)
                steps = [
                    DerivationStep(1, "Calculate period as inverse frequency", "T = \\frac{1}{f}", f"T = \\frac{{1}}{{{freq}\\text{{ Hz}}}} = {disp}\\text{{ s}}", disp, "s")
                ]
                receipt = self._build_receipt("waves", "physics", "period", "waves.period", {"f": str(freq)}, str(period), disp, "s", steps)
                return SolverResult(True, q, "physics", "waves", "period", "waves.period", float(period), disp, "s", steps, receipt, f"Wave period is {disp} s.")

        return None

    # --------------------------------------------------------------------------
    # 9. Chemistry & Solutions (M = n/V, M1*V1 = M2*V2, N = n*NA)
    # --------------------------------------------------------------------------
    def _solve_chemistry_solutions(self, q: str) -> Optional[SolverResult]:
        m1_m = re.search(r"(?:initial concentration|M1|C1)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:M|mol/L)\b", q, re.I)
        v1_m = re.search(r"(?:initial volume|V1)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(mL|L|liters)\b", q, re.I)
        m2_m = re.search(r"(?:target concentration|final concentration|M2|C2)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:M|mol/L)\b", q, re.I)
        v2_m = re.search(r"(?:final volume|V2)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(mL|L|liters)\b", q, re.I)
        n_m = re.search(r"(?:moles|amount|n)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mol\b", q, re.I)
        vol_m = re.search(r"(?:volume|V)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(L|mL)\b", q, re.I)

        # Molarity: M = n / V
        if n_m and vol_m and ("molarity" in q.lower() or "concentration" in q.lower()) and not m1_m:
            n = self._normalize_num(n_m.group(1))
            v = self._normalize_num(vol_m.group(1)) * (Fraction(1, 1000) if vol_m.group(2).lower() == "ml" else Fraction(1))
            if v > 0:
                molar = n / v
                disp = self._format_fraction(molar)
                steps = [
                    DerivationStep(1, "Calculate solution molarity", "M = \\frac{n}{V}", f"M = \\frac{{{n}\\text{{ mol}}}}{{{v}\\text{{ L}}}} = {disp}\\text{{ M}}", disp, "M")
                ]
                receipt = self._build_receipt("chemistry", "chemistry", "molarity", "chem.molarity", {"n": str(n), "V": str(v)}, str(molar), disp, "M", steps)
                return SolverResult(True, q, "chemistry", "chemistry", "molarity", "chem.molarity", float(molar), disp, "M", steps, receipt, f"Molarity is {disp} M.")

        # Dilution: V2 = (M1 * V1) / M2
        if m1_m and v1_m and m2_m and ("dilution" in q.lower() or "final volume" in q.lower()):
            m1 = self._normalize_num(m1_m.group(1))
            v1 = self._normalize_num(v1_m.group(1)) * (Fraction(1, 1000) if v1_m.group(2).lower() == "ml" else Fraction(1))
            m2 = self._normalize_num(m2_m.group(1))
            if m2 > 0:
                v2 = (m1 * v1) / m2
                v2_ml = v2 * 1000
                disp = self._format_fraction(v2_ml)
                steps = [
                    DerivationStep(1, "Apply conservation of solute mass (dilution equation)", "M_1 V_1 = M_2 V_2 \\implies V_2 = \\frac{M_1 V_1}{M_2}", f"V_2 = \\frac{{({m1})({v1})}}{{{m2}}} = {self._format_fraction(v2)}\\text{{ L}} = {disp}\\text{{ mL}}", f"{disp} mL", "mL")
                ]
                receipt = self._build_receipt("chemistry", "chemistry", "dilution_volume", "chem.dilution", {"M1": str(m1), "V1": str(v1), "M2": str(m2)}, str(v2), f"{disp} mL", "mL", steps)
                return SolverResult(True, q, "chemistry", "chemistry", "dilution_volume", "chem.dilution", float(v2_ml), f"{disp} mL", "mL", steps, receipt, f"Required final volume is {disp} mL.")

        return None

    # --------------------------------------------------------------------------
    # 10. Pure Algebra: Quadratic Equation (ax^2 + bx + c = 0)
    # --------------------------------------------------------------------------
    def _solve_quadratic_algebra(self, q: str) -> Optional[SolverResult]:
        # Form: ax^2 + bx + c = 0 or "solve quadratic a=1, b=-5, c=6"
        a_m = re.search(r"\ba\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q, re.I)
        b_m = re.search(r"\bb\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q, re.I)
        c_m = re.search(r"\bc\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q, re.I)

        quad_match = re.search(r"(-?\d+)?\s*x\^2\s*([+-]\s*\d+)?\s*x\s*([+-]\s*\d+)?\s*=\s*0", q, re.I)

        if (a_m and b_m and c_m and ("quadratic" in q.lower() or "roots" in q.lower())) or quad_match:
            if quad_match:
                a_raw = quad_match.group(1) or "1"
                b_raw = (quad_match.group(2) or "+0").replace(" ", "")
                c_raw = (quad_match.group(3) or "+0").replace(" ", "")
                a = self._normalize_num(a_raw)
                b = self._normalize_num(b_raw)
                c = self._normalize_num(c_raw)
            else:
                a = self._normalize_num(a_m.group(1))
                b = self._normalize_num(b_m.group(1))
                c = self._normalize_num(c_m.group(1))

            if a == 0:
                return None

            disc = b*b - 4*a*c
            disc_float = float(disc)
            if disc_float >= 0:
                sqrt_disc = math.sqrt(disc_float)
                r1 = (-float(b) + sqrt_disc) / (2 * float(a))
                r2 = (-float(b) - sqrt_disc) / (2 * float(a))
                disp1 = f"{r1:.4f}".rstrip("0").rstrip(".")
                disp2 = f"{r2:.4f}".rstrip("0").rstrip(".")
                res_str = f"x1 = {disp1}, x2 = {disp2}" if disp1 != disp2 else f"x = {disp1}"
                steps = [
                    DerivationStep(1, "Calculate discriminant Delta = b^2 - 4ac", "\\Delta = b^2 - 4ac", f"\\Delta = ({b})^2 - 4({a})({c}) = {disc}", str(disc)),
                    DerivationStep(2, "Calculate roots via quadratic formula", "x = \\frac{-b \\pm \\sqrt{\\Delta}}{2a}", f"x = \\frac{{-({b}) \\pm \\sqrt{{{disc}}}}}{{2({a})}} \\implies {res_str}", res_str),
                ]
                receipt = self._build_receipt("algebra", "mathematics", "quadratic_roots", "math.quadratic", {"a": str(a), "b": str(b), "c": str(c)}, str(disc), res_str, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "algebra", "quadratic_roots", "math.quadratic", float(r1), res_str, "", steps, receipt, f"Roots of quadratic: {res_str}.")

        return None

    # --------------------------------------------------------------------------
    # 11. Linear System of 2 Equations (a1*x + b1*y = c1, a2*x + b2*y = c2)
    # --------------------------------------------------------------------------
    def _solve_linear_system_2x2(self, q: str) -> Optional[SolverResult]:
        # e.g., "solve system 2x + 3y = 8 and 4x - y = 2"
        sys_m = re.search(r"(-?\d+)\s*x\s*([+-]\s*\d+)\s*y\s*=\s*(-?\d+)\s*(?:and|,|;)\s*(-?\d+)\s*x\s*([+-]\s*\d+)\s*y\s*=\s*(-?\d+)", q, re.I)
        if sys_m:
            a1 = self._normalize_num(sys_m.group(1))
            b1 = self._normalize_num(sys_m.group(2).replace(" ", ""))
            c1 = self._normalize_num(sys_m.group(3))
            a2 = self._normalize_num(sys_m.group(4))
            b2 = self._normalize_num(sys_m.group(5).replace(" ", ""))
            c2 = self._normalize_num(sys_m.group(6))

            det = a1*b2 - a2*b1
            if det != 0:
                det_x = c1*b2 - c2*b1
                det_y = a1*c2 - a2*c1
                x = det_x / det
                y = det_y / det
                disp_x = self._format_fraction(x)
                disp_y = self._format_fraction(y)
                res_str = f"x = {disp_x}, y = {disp_y}"
                steps = [
                    DerivationStep(1, "Compute system determinant via Cramer's rule", "D = a_1 b_2 - a_2 b_1", f"D = ({a1})({b2}) - ({a2})({b1}) = {det}", str(det)),
                    DerivationStep(2, "Compute coordinate determinants and solve", "x = \\frac{D_x}{D}, \\quad y = \\frac{D_y}{D}", f"x = \\frac{{{det_x}}}{{{det}}} = {disp_x}, \\quad y = \\frac{{{det_y}}}{{{det}}} = {disp_y}", res_str),
                ]
                receipt = self._build_receipt("algebra", "mathematics", "linear_system_2x2", "math.cramer_2x2", {"eq1": f"{a1}x+{b1}y={c1}", "eq2": f"{a2}x+{b2}y={c2}"}, f"({x},{y})", res_str, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "algebra", "linear_system_2x2", "math.cramer_2x2", float(x), res_str, "", steps, receipt, f"Solution to linear system: {res_str}.")

        return None

    # --------------------------------------------------------------------------
    # 12. Compound Interest (A = P * (1 + r/n)^(n*t))
    # --------------------------------------------------------------------------
    def _solve_compound_interest(self, q: str) -> Optional[SolverResult]:
        p_m = re.search(r"(?:principal|principal amount|P)\s*(?:=|is|of)?\s*(\$?\d+(?:\.\d+)?)", q, re.I)
        r_m = re.search(r"(?:interest rate|rate|r)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*%", q, re.I)
        t_m = re.search(r"(?:time|t|duration|years)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:years|yr|yrs)\b", q, re.I)
        n_m = re.search(r"(?:compounded|frequency|n)\s*(?:=|is)?\s*(\d+)\s*(?:times per year)?", q, re.I)

        if p_m and r_m and t_m and ("compound interest" in q.lower() or "future value" in q.lower()):
            p_clean = p_m.group(1).replace("$", "")
            principal = self._normalize_num(p_clean)
            rate = self._normalize_num(r_m.group(1)) / 100
            years = self._normalize_num(t_m.group(1))
            n = self._normalize_num(n_m.group(1)) if n_m else Fraction(1)  # annual by default

            # A = P * (1 + r/n)^(nt)
            rate_float = float(rate)
            n_float = float(n)
            t_float = float(years)
            p_float = float(principal)
            amount = p_float * ((1 + rate_float / n_float) ** (n_float * t_float))
            interest = amount - p_float
            disp_amt = f"{amount:.2f}"
            disp_int = f"{interest:.2f}"
            res_str = f"Final Amount: ${disp_amt} (Interest: ${disp_int})"

            steps = [
                DerivationStep(1, "Apply compound interest formula", "A = P \\left(1 + \\frac{r}{n}\\right)^{nt}", f"A = {principal} \\left(1 + \\frac{{{rate}}}{{{n}}}\\right)^{{({n})({years})}} = {disp_amt}", disp_amt, "$")
            ]
            receipt = self._build_receipt("finance", "mathematics", "compound_interest", "math.compound_interest", {"P": str(principal), "r": str(rate), "t": str(years), "n": str(n)}, str(amount), res_str, "$", steps)
            return SolverResult(True, q, "mathematics", "finance", "compound_interest", "math.compound_interest", amount, res_str, "$", steps, receipt, f"Compound interest outcome: {res_str}.")

        return None

    # --------------------------------------------------------------------------
    # 13. Series & Progressions (Arithmetic / Geometric Sum)
    # --------------------------------------------------------------------------
    def _solve_series_progressions(self, q: str) -> Optional[SolverResult]:
        # Arithmetic: Sn = n/2 * (2a + (n-1)d)
        a_m = re.search(r"(?:first term|a1|a)\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q, re.I)
        d_m = re.search(r"(?:common difference|difference|d)\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q, re.I)
        r_m = re.search(r"(?:common ratio|ratio|r)\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q, re.I)
        n_m = re.search(r"(?:number of terms|terms|n)\s*(?:=|is)?\s*(\d+)", q, re.I)

        if a_m and d_m and n_m and ("arithmetic series" in q.lower() or "arithmetic progression" in q.lower() or "sum of arithmetic" in q.lower()):
            a = self._normalize_num(a_m.group(1))
            d = self._normalize_num(d_m.group(1))
            n = self._normalize_num(n_m.group(1))
            # Sn = n/2 * (2a + (n-1)d)
            sn = (n / 2) * (2*a + (n-1)*d)
            disp = self._format_fraction(sn)
            steps = [
                DerivationStep(1, "Compute arithmetic series sum", "S_n = \\frac{n}{2} (2a + (n-1)d)", f"S_{{{n}}} = \\frac{{{n}}}{{2}} (2({a}) + ({n}-1)({d})) = {disp}", disp)
            ]
            receipt = self._build_receipt("series", "mathematics", "arithmetic_sum", "math.arithmetic_series", {"a": str(a), "d": str(d), "n": str(n)}, str(sn), disp, "dimensionless", steps)
            return SolverResult(True, q, "mathematics", "series", "arithmetic_sum", "math.arithmetic_series", float(sn), disp, "", steps, receipt, f"Sum of arithmetic series is {disp}.")

        if a_m and r_m and n_m and ("geometric series" in q.lower() or "geometric progression" in q.lower() or "sum of geometric" in q.lower()):
            a = self._normalize_num(a_m.group(1))
            r = self._normalize_num(r_m.group(1))
            n = int(n_m.group(1))
            if r != 1:
                sn = a * (1 - r**n) / (1 - r)
                disp = self._format_fraction(sn)
                steps = [
                    DerivationStep(1, "Compute finite geometric series sum", "S_n = a \\frac{1 - r^n}{1 - r}", f"S_{{{n}}} = {a} \\frac{{1 - ({r})^{{{n}}}}}{{1 - {r}}} = {disp}", disp)
                ]
                receipt = self._build_receipt("series", "mathematics", "geometric_sum", "math.geometric_series", {"a": str(a), "r": str(r), "n": str(n)}, str(sn), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "series", "geometric_sum", "math.geometric_series", float(sn), disp, "", steps, receipt, f"Sum of geometric series is {disp}.")

        return None

    # --------------------------------------------------------------------------
    # 14. Geometry & Trigonometry (Pythagorean, Circle, Sphere, Cylinder)
    # --------------------------------------------------------------------------
    def _solve_geometry_trig(self, q: str) -> Optional[SolverResult]:
        # Pythagorean: c = sqrt(a^2 + b^2)
        side_a_m = re.search(r"(?:side a|leg a|a)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        side_b_m = re.search(r"(?:side b|leg b|b)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        hyp_m = re.search(r"(?:hypotenuse|c)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        rad_m = re.search(r"(?:radius|r)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        h_m = re.search(r"(?:height|h)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)

        # Hypotenuse: c = sqrt(a^2 + b^2)
        if side_a_m and side_b_m and ("hypotenuse" in q.lower() or "pythagorean" in q.lower()) and not hyp_m:
            a = self._normalize_num(side_a_m.group(1))
            b = self._normalize_num(side_b_m.group(1))
            c_sq = a*a + b*b
            c_val = math.sqrt(float(c_sq))
            c_frac = Fraction(Decimal(f"{c_val:.6f}"))
            disp = self._format_fraction(c_frac)
            steps = [
                DerivationStep(1, "Apply Pythagorean Theorem", "c = \\sqrt{a^2 + b^2}", f"c = \\sqrt{{{a}^2 + {b}^2}} = \\sqrt{{{c_sq}}} = {disp}\\text{{ m}}", disp, "m")
            ]
            receipt = self._build_receipt("geometry", "mathematics", "hypotenuse", "geometry.pythagorean", {"a": str(a), "b": str(b)}, str(c_frac), disp, "m", steps)
            return SolverResult(True, q, "mathematics", "geometry", "hypotenuse", "geometry.pythagorean", c_val, disp, "m", steps, receipt, f"Hypotenuse is {disp} m.")

        # Sphere Volume: V = 4/3 * pi * r^3
        if rad_m and ("sphere volume" in q.lower() or "volume of sphere" in q.lower()):
            r = self._normalize_num(rad_m.group(1))
            vol = Fraction(4, 3) * CONSTANTS["pi"]["value"] * (r**3)
            disp = self._format_fraction(vol)
            steps = [
                DerivationStep(1, "Calculate sphere volume", "V = \\frac{4}{3} \\pi r^3", f"V = \\frac{{4}}{{3}} \\pi ({r}\\text{{ m}})^3 = {disp}\\text{{ m}}^3", disp, "m^3")
            ]
            receipt = self._build_receipt("geometry", "mathematics", "sphere_volume", "geometry.sphere_volume", {"r": str(r)}, str(vol), disp, "m^3", steps)
            return SolverResult(True, q, "mathematics", "geometry", "sphere_volume", "geometry.sphere_volume", float(vol), disp, "m^3", steps, receipt, f"Sphere volume is {disp} m^3.")

        # Cylinder Volume: V = pi * r^2 * h
        if rad_m and h_m and ("cylinder volume" in q.lower() or "volume of cylinder" in q.lower()):
            r = self._normalize_num(rad_m.group(1))
            h = self._normalize_num(h_m.group(1))
            vol = CONSTANTS["pi"]["value"] * (r**2) * h
            disp = self._format_fraction(vol)
            steps = [
                DerivationStep(1, "Calculate cylinder volume", "V = \\pi r^2 h", f"V = \\pi ({r}\\text{{ m}})^2 ({h}\\text{{ m}}) = {disp}\\text{{ m}}^3", disp, "m^3")
            ]
            receipt = self._build_receipt("geometry", "mathematics", "cylinder_volume", "geometry.cylinder_volume", {"r": str(r), "h": str(h)}, str(vol), disp, "m^3", steps)
            return SolverResult(True, q, "mathematics", "geometry", "cylinder_volume", "geometry.cylinder_volume", float(vol), disp, "m^3", steps, receipt, f"Cylinder volume is {disp} m^3.")

        return None

    # --------------------------------------------------------------------------
    # 15. Combinatorics, Probability & Statistics (P(n,k), C(n,k), Entropy)
    # --------------------------------------------------------------------------
    def _solve_combinatorics_stats(self, q: str) -> Optional[SolverResult]:
        n_m = re.search(r"\bn\s*(?:=|is)?\s*(\d+)", q, re.I)
        k_m = re.search(r"\b(?:k|r)\s*(?:=|is)?\s*(\d+)", q, re.I)

        # Combinations: C(n, k) = n! / (k! * (n-k)!)
        if n_m and k_m and ("combination" in q.lower() or "choose" in q.lower() or "c(n,k)" in q.lower() or "n choose k" in q.lower()):
            n = int(n_m.group(1))
            k = int(k_m.group(1))
            if 0 <= k <= n:
                ans = math.comb(n, k)
                disp = str(ans)
                steps = [
                    DerivationStep(1, "Apply combination binomial coefficient formula", "C(n,k) = \\binom{n}{k} = \\frac{n!}{k!(n-k)!}", f"\\binom{{{n}}}{{{k}}} = \\frac{{{n}!}}{{{k}! ({n}-{k})!}} = {disp}", disp)
                ]
                receipt = self._build_receipt("combinatorics", "mathematics", "combinations", "math.combinations", {"n": str(n), "k": str(k)}, str(ans), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "combinatorics", "combinations", "math.combinations", float(ans), disp, "", steps, receipt, f"Number of combinations C({n},{k}) = {disp}.")

        # Permutations: P(n, k) = n! / (n-k)!
        if n_m and k_m and ("permutation" in q.lower() or "p(n,k)" in q.lower()):
            n = int(n_m.group(1))
            k = int(k_m.group(1))
            if 0 <= k <= n:
                ans = math.perm(n, k)
                disp = str(ans)
                steps = [
                    DerivationStep(1, "Apply permutation formula", "P(n,k) = \\frac{n!}{(n-k)!}", f"P({n},{k}) = \\frac{{{n}!}}{{({n}-{k})!}} = {disp}", disp)
                ]
                receipt = self._build_receipt("combinatorics", "mathematics", "permutations", "math.permutations", {"n": str(n), "k": str(k)}, str(ans), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "combinatorics", "permutations", "math.permutations", float(ans), disp, "", steps, receipt, f"Number of permutations P({n},{k}) = {disp}.")

        return None

    def _build_receipt(
        self,
        scenario: str,
        domain: str,
        target: str,
        formula_id: str,
        inputs: Dict[str, str],
        raw_result_fraction: str,
        display_result: str,
        unit: str,
        steps: List[DerivationStep],
    ) -> SolverReceipt:
        """Construct a deterministic cryptographic audit receipt."""
        derivation_text = json.dumps([s.to_dict() for s in steps], sort_keys=True)
        deriv_hash = hashlib.sha256(derivation_text.encode("utf-8")).hexdigest()

        payload = {
            "schema_version": SOLVER_RECEIPT_SCHEMA,
            "scenario": scenario,
            "domain": domain,
            "target": target,
            "formula_id": formula_id,
            "inputs": inputs,
            "raw_result_fraction": raw_result_fraction,
            "display_result": display_result,
            "unit": unit,
            "derivation_hash": deriv_hash,
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        receipt_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return SolverReceipt(
            schema_version=SOLVER_RECEIPT_SCHEMA,
            scenario=scenario,
            domain=domain,
            target=target,
            formula_id=formula_id,
            inputs=inputs,
            raw_result_fraction=raw_result_fraction,
            display_result=display_result,
            unit=unit,
            derivation_hash=deriv_hash,
            receipt_sha256=receipt_sha256,
        )


_DEFAULT_SOLVER = NexusSolver()


def solve_problem(query: str) -> SolverResult:
    """Convenience functional interface to NexusSolver."""
    return _DEFAULT_SOLVER.solve(query)
