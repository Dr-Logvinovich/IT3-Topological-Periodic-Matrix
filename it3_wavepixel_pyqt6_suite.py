#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IT3-WavePixel v3.1  ::  Hardened Topological Cascade & Transmutation Suite
Authors: Victor Logvinovich (ORCID 0009-0007-2040-6955), Kobrin, Belarus
         Jean-Claude Perez  (ORCID 0000-0001-6446-2042), Montpellier

Epistemic protocol:  A = theorem, B = conditional, C = resonance, O = open.
No hardcoded "N/N passed" banner: the ledger counts its own checks at runtime.

Run:
    python3 it3_wavepixel_pyqt6_suite.py                 # GUI (PyQt6)
    python3 it3_wavepixel_pyqt6_suite.py --self-test     # Level-A ledger + firewall
    python3 it3_wavepixel_pyqt6_suite.py --transmute 208Pb 197Au
    python3 it3_wavepixel_pyqt6_suite.py --batch-export out/
    python3 it3_wavepixel_pyqt6_suite.py --ledger ledger.json
"""
from __future__ import annotations

import sys, os, re, math, csv, json, hashlib, argparse, itertools
from dataclasses import dataclass, asdict
from fractions import Fraction
from functools import lru_cache
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import matplotlib

VERSION = "3.1"
_HEADLESS = any(f in sys.argv for f in
                ("--self-test", "--transmute", "--batch-export", "--ledger", "--firewall"))
if _HEADLESS:
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("QtAgg")
    except Exception:
        matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QGridLayout, QLabel, QSpinBox, QComboBox,
                                 QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
                                 QGroupBox, QHeaderView, QFileDialog, QMessageBox, QSlider,
                                 QSplitter, QFrame, QTextEdit, QLineEdit, QScrollArea)
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QColor
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

BG, PANEL, GRID, TXT = "#0B0E17", "#121726", "#232B42", "#E0E6ED"
CYAN, PINK, AMBER, GREEN = "#00E5FF", "#FF2A85", "#FFC400", "#00E676"
BLOCK_COLORS = {"s": "#FFB300", "f": "#00E5FF", "d": "#FF1744", "p": "#B388FF"}


# ============================================================================
# 0. LEDGER
# ============================================================================
@dataclass
class CheckRow:
    level: str
    name: str
    passed: bool
    expected: bool
    detail: str = ""


class Ledger:
    def __init__(self) -> None:
        self.rows: List[CheckRow] = []

    def check(self, level: str, name: str, fn: Callable[[], bool],
              expect: bool = True, note: str = "") -> bool:
        try:
            got = bool(fn())
            ok = (got == expect)
            detail = note or f"got={got} expected={expect}"
        except Exception as exc:
            ok, detail = False, f"EXCEPTION {type(exc).__name__}: {exc}"
        self.rows.append(CheckRow(level, name, ok, expect, detail))
        return ok

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def n_pass(self) -> int:
        return sum(r.passed for r in self.rows)

    def digest(self) -> str:
        payload = json.dumps([asdict(r) for r in self.rows], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def report(self) -> None:
        for r in self.rows:
            print(f"  [{r.level:<7}] {'PASS' if r.passed else 'FAIL'}  {r.name}")
            if not r.passed:
                print(f"          -> {r.detail}")
        print(f"\n  LEDGER: {self.n_pass}/{self.n} passed (count computed at runtime)")
        print(f"  SHA256: {self.digest()}")

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": VERSION, "n": self.n, "passed": self.n_pass,
                       "sha256": self.digest(), "rows": [asdict(r) for r in self.rows]},
                      fh, indent=2, ensure_ascii=False)


class RegisterOverflow(ValueError):
    """Z outside the 172-state register."""


class PhysicalViolation(ValueError):
    """Unphysical state: A<Z, N<0, alpha with N<2."""


# ============================================================================
# 1. 172-LAYER MATRIX
# ============================================================================
class WavePixelMatrix:
    N_MAX = 172
    CAPACITIES: Tuple[int, ...] = tuple(2 * ((min(n, 11 - n) + 3) // 2) ** 2 for n in range(12))
    CLOSURES: Tuple[int, ...] = tuple(
        sum(2 * ((min(k, 11 - k) + 3) // 2) ** 2 for k in range(n + 1)) for n in range(12))
    DIODE_CLOSURES: Tuple[int, ...] = tuple(e for e in CLOSURES if e % 8 == 6)
    VACUUM_MASK = 0xE100
    KUMMER_WALL = 13

    BLOCK_SEQ: Dict[int, Tuple[Tuple[str, int], ...]] = {
        2: (("s", 2),),
        8: (("s", 2), ("p", 6)),
        18: (("s", 2), ("d", 10), ("p", 6)),
        32: (("s", 2), ("f", 14), ("d", 10), ("p", 6))}

    SYMBOLS: Dict[int, str] = {
        1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",11:"Na",12:"Mg",
        13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",
        24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",
        34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",42:"Mo",43:"Tc",
        44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",51:"Sb",52:"Te",53:"I",
        54:"Xe",55:"Cs",56:"Ba",57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",
        64:"Gd",65:"Tb",66:"Dy",67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu",72:"Hf",73:"Ta",
        74:"W",75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",81:"Tl",82:"Pb",83:"Bi",
        84:"Po",85:"At",86:"Rn",87:"Fr",88:"Ra",89:"Ac",90:"Th",91:"Pa",92:"U",93:"Np",
        94:"Pu",95:"Am",96:"Cm",97:"Bk",98:"Cf",99:"Es",100:"Fm",101:"Md",102:"No",103:"Lr",
        104:"Rf",105:"Db",106:"Sg",107:"Bh",108:"Hs",109:"Mt",110:"Ds",111:"Rg",112:"Cn",
        113:"Nh",114:"Fl",115:"Mc",116:"Lv",117:"Ts",118:"Og",119:"Uue",120:"Ubn",121:"Ubu",
        122:"Ubb",123:"Ubt",124:"Ubq",125:"Ubp",126:"Ubh"}

    NAMES: Dict[int, str] = {
        1:"Hydrogen",2:"Helium",6:"Carbon",8:"Oxygen",11:"Sodium",17:"Chlorine",18:"Argon",
        26:"Iron",36:"Krypton",43:"Technetium",54:"Xenon",55:"Cesium",56:"Barium",
        70:"Ytterbium",78:"Platinum",79:"Gold",80:"Mercury",81:"Thallium",82:"Lead",
        83:"Bismuth",84:"Polonium",85:"Astatine",86:"Radon",87:"Francium",88:"Radium",
        90:"Thorium",92:"Uranium",94:"Plutonium",96:"Curium",98:"Californium",102:"Nobelium",
        103:"Lawrencium",104:"Rutherfordium",105:"Dubnium",110:"Darmstadtium",
        111:"Roentgenium",112:"Copernicium",113:"Nihonium",114:"Flerovium",115:"Moscovium",
        116:"Livermorium",117:"Tennessine",118:"Oganesson",119:"Ununennium",120:"Unbinilium",
        126:"Unbihexium",136:"Untrihexium",137:"Untriseptium",172:"Unseptbium"}

    STANDARD_MASS: Dict[int, int] = {
        1:1,2:4,3:7,4:9,5:11,6:12,7:14,8:16,9:19,10:20,11:23,12:24,13:27,14:28,15:31,
        16:32,17:35,18:40,19:39,20:40,26:56,29:63,30:64,36:84,43:98,47:107,50:118,54:132,
        55:133,56:138,70:174,74:184,78:195,79:197,80:202,81:205,82:208,83:209,84:210,
        85:210,86:222,87:223,88:226,89:227,90:232,91:231,92:238,93:237,94:244,95:243,
        96:250,97:247,98:252,99:252,100:257,101:258,102:259,103:262,104:267,105:268,
        106:271,107:272,108:270,109:276,110:281,111:280,112:285,113:286,114:289,115:290,
        116:293,117:294,118:294,119:297,120:298,126:310,136:336,137:340,172:426}

    @classmethod
    def get_symbol(cls, z: int) -> str:
        return cls.SYMBOLS.get(z, f"E{z}")

    @classmethod
    def cell_labels(cls, z: int) -> Tuple[str, str]:
        return (cls.SYMBOLS[z], str(z)) if z in cls.SYMBOLS else (str(z), "")

    @classmethod
    def get_name(cls, z: int) -> str:
        return cls.NAMES.get(z, f"Element-{z}")

    @classmethod
    def floor_of(cls, z: int) -> int:
        if not (1 <= z <= cls.N_MAX):
            raise RegisterOverflow(f"Z={z} outside the 172-layer register [1..172]")
        for n, s in enumerate(cls.CLOSURES):
            if z <= s:
                return n
        return 11

    @classmethod
    def get_coordinates(cls, z: int) -> Tuple[int, str, int, int]:
        n = cls.floor_of(z)
        prev = cls.CLOSURES[n - 1] if n > 0 else 0
        mu = z - prev
        acc = 0
        for blk, cap in cls.BLOCK_SEQ[cls.CAPACITIES[n]]:
            if mu <= acc + cap:
                return n, blk, mu, mu - acc
            acc += cap
        raise RegisterOverflow(f"coordinate overflow at Z={z}")

    @classmethod
    def facing_partner(cls, z: int) -> int:
        n = cls.floor_of(z)
        prev = cls.CLOSURES[n - 1] if n > 0 else 0
        return z + 172 - cls.CLOSURES[n] - prev

    @classmethod
    def nuclear_facing_word(cls, z: int) -> int:
        return (z << 8) | cls.facing_partner(z)

    @classmethod
    def closure_distance(cls, z: int) -> int:
        return min(abs(z - s) for s in cls.CLOSURES)

    @classmethod
    def reference_mass(cls, z: int) -> int:
        """UI initialisation only; not a valley fit."""
        return cls.STANDARD_MASS.get(z, 2 * z)

    @classmethod
    def mask_int(cls, z: int) -> int:
        if not (0 <= z <= cls.N_MAX):
            raise RegisterOverflow("mask address must be in 0..172")
        return (1 << z) - 1

    @classmethod
    def parse_nuclide_input(cls, text: str) -> Tuple[int, int]:
        raw = text.strip()
        if not raw:
            raise ValueError("Empty nuclide input.")
        if "," in raw or ":" in raw:
            parts = re.split(r"[,:]", raw)
            if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                raise ValueError(f"Invalid Z,A pair '{text}'.")
            z, a = int(parts[0]), int(parts[1])
            if not (1 <= z <= 172):
                raise ValueError(f"Z={z} out of range 1..172.")
            if a < z:
                raise ValueError(f"Invalid mass A={a} < Z={z}.")
            return z, a
        sym = {s.lower(): z for z, s in cls.SYMBOLS.items()}
        nam = {n.lower(): z for z, n in cls.NAMES.items()}
        for z in range(1, 173):
            sym[f"e{z}"] = z
        m = re.match(r"^(\d+)\s*[-_]?\s*([A-Za-z]+\d*)$", raw)
        if m:
            a, key = int(m.group(1)), m.group(2).lower()
            z = sym.get(key) or nam.get(key)
            if z is not None:
                if a < z:
                    raise ValueError(f"Invalid mass A={a} < Z={z}.")
                return z, a
        m = re.match(r"^([A-Za-z]+\d*)\s*[-_\s]\s*(\d+)$", raw)
        if not m:
            m = re.match(r"^([A-Za-z]+)(\d+)$", raw)
        if m and m.group(1).lower() == "e":
            m = None
        if m:
            key, a = m.group(1).lower(), int(m.group(2))
            z = sym.get(key) or nam.get(key)
            if z is not None:
                if a < z:
                    raise ValueError(f"Invalid mass A={a} < Z={z}.")
                return z, a
        raise ValueError(f"Rejected '{text}': explicit A >= Z required "
                         f"(e.g. '208Pb', 'Au-197', '82,208').")


WPM = WavePixelMatrix


# ============================================================================
# 2. THREE-REGISTER STATE
# ============================================================================
@dataclass
class ThreeRegisterState:
    Z: int
    N: int
    M_weight: int

    def __post_init__(self) -> None:
        if self.Z < 0 or self.N < 0:
            raise PhysicalViolation(f"negative register Z={self.Z} N={self.N}")

    @property
    def A(self) -> int:
        return self.Z + self.N

    @property
    def Q_top(self) -> int:
        return self.Z - self.M_weight

    @property
    def strain_proxy(self) -> Fraction:
        if self.A <= 0:
            raise PhysicalViolation("strain requires A>0")
        return Fraction(self.Z * self.Z, self.A)

    @property
    def asymmetry_proxy(self) -> Fraction:
        if self.A <= 0:
            raise PhysicalViolation("asymmetry requires A>0")
        return Fraction(self.N - self.Z, self.A)


# ============================================================================
# 3. BINARY FRICTION ALU
# ============================================================================
def wt(x: int) -> int:
    return bin(x).count("1")


def nu2(m: int) -> int:
    if m <= 0:
        return 0
    t = 0
    while (m & 1) == 0:
        m >>= 1
        t += 1
    return t


class BinaryFrictionALU:
    popcount = staticmethod(wt)
    nu2 = staticmethod(nu2)

    @classmethod
    def compute_tfi(cls, zp: int, zd: int) -> Tuple[int, int]:
        mask = zp ^ zd
        return mask, wt(mask)

    @staticmethod
    def tfi_pow2(z: int, j: int) -> int:
        """Theorem D: wt(Z ^ (Z-2^j)) = nu2(Z>>j)+1."""
        return wt(z ^ (z - (1 << j)))

    @classmethod
    def compute_atfi_sq(cls, zp: int, zd: int) -> Tuple[int, Dict[str, int]]:
        d = {"s": 0, "p": 0, "d": 0, "f": 0}
        if zp > zd:
            for z in range(zp, zd, -1):
                d[WPM.get_coordinates(z)[1]] -= 1
        elif zp < zd:
            for z in range(zp + 1, zd + 1):
                d[WPM.get_coordinates(z)[1]] += 1
        return d["s"] ** 2 + 2 * d["p"] ** 2 + 3 * d["d"] ** 2, d

    @classmethod
    def compute_atfi(cls, zp: int, zd: int) -> Tuple[float, Dict[str, int]]:
        sq, d = cls.compute_atfi_sq(zp, zd)
        return math.sqrt(sq), d

    @staticmethod
    def audit_nibble_borrow(z1: int, z2: int) -> str:
        h1, l1 = (z1 >> 4) & 0xF, z1 & 0xF
        h2, l2 = (z2 >> 4) & 0xF, z2 & 0xF
        dz = z2 - z1
        if dz < 0:
            k = -dz
            kh, kl = (k >> 4) & 0xF, k & 0xF
            if l1 < kl:
                return (f"Low-nibble borrow: 0x{l1:X}<0x{kl:X}, high 0x{h1:X}->0x{h2:X}; "
                        f"0x{z1:02X}-0x{k:02X}=0x{z2:02X}")
            if h2 < h1:
                return f"High-nibble shift, no low borrow (0x{h1:X}-0x{kh:X}->0x{h2:X})"
            return f"Local low-nibble subtraction 0x{l1:X}->0x{l2:X}"
        if dz > 0:
            kl = dz & 0xF
            if l1 + kl >= 16:
                return f"Low-nibble carry: 0x{l1:X}+0x{kl:X} (high 0x{h1:X}->0x{h2:X})"
            if h2 > h1:
                return f"High-nibble shift, no low carry (0x{h1:X}->0x{h2:X})"
            return f"Local low-nibble addition 0x{l1:X}->0x{l2:X}"
        return "Identity address"


def nibble_closed(N: int, b: int = 16) -> Tuple[int, int, int]:
    """Theorem E: closed forms of the borrow partition for any N, any base."""
    H, L = divmod(N, b)
    S = b * (b - 1) // 2
    borrow = S * H * (H - 1) // 2 + H * ((b - 1) * (L + 1) - L * (L + 1) // 2)
    high = (S + b) * H * (H - 1) // 2 + H * (L + 1) * (L + 2) // 2
    quiet = S * H + L * (L + 1) // 2
    return borrow, high, quiet


def nibble_brute(N: int, b: int = 16) -> Tuple[int, int, int, int]:
    bo = hi = qu = fp = 0
    for z in range(1, N + 1):
        h, l = divmod(z, b)
        for k in range(1, z + 1):
            hk, m = divmod(k, b)
            borrow = 1 if l < m else 0
            if h - hk - borrow != (z - k) // b:
                raise AssertionError("nibble borrow identity violated")
            if borrow:
                bo += 1
            elif hk:
                hi += 1
                if k < z:
                    fp += 1
            else:
                qu += 1
    return bo, hi, qu, fp


# ============================================================================
# 4. CASCADE ENGINE
# ============================================================================
@dataclass
class DecayStepRecord:
    step_idx: int
    mode: str
    z_parent: int
    a_parent: int
    z_daughter: int
    a_daughter: int
    bin_parent: str
    bin_daughter: str
    xor_mask: int
    bin_xor: str
    tfi: int
    atfi: float
    coord_parent: str
    coord_daughter: str
    block_parent: str
    block_daughter: str
    floor_parent: int
    floor_daughter: int
    facing_word_hex: str
    strain_before: float
    strain_after: float
    asym_after: float
    q_top_unrelaxed: int
    event_tag: str
    tunnel_hint: str


@dataclass
class ChainResult:
    records: List[DecayStepRecord]
    halt: str = ""
    requested_stop: int = 0
    final_z: int = 0
    final_a: int = 0


@dataclass
class CustomTransmutationResult:
    z1: int; a1: int; n1: int
    z2: int; a2: int; n2: int
    dz: int; dn: int; da: int
    xor_addr: int; tfi_addr: int
    atfi_total: float; delta_blocks: Dict[str, int]
    nibble_report: str; q_top_unrelaxed: int; mask_xor_weight: int
    strain1: float; strain2: float; asym1: float; asym2: float
    direct_equation: str; stepwise_records: List[DecayStepRecord]
    atfi_sq: int = 0
    nucleon_splits: Tuple[Tuple[int, int, int], ...] = ()
    alpha_ec_reachable: bool = False
    alpha_ec_moves: Tuple[int, int] = (0, 0)
    reachability_note: str = ""
    path_valid: bool = True
    path_note: str = ""


class AlphaCascadeEngine:
    @staticmethod
    def cascade_moves(zp: int, ap: int, zd: int, ad: int) -> Tuple[bool, int, int, str]:
        dz, da = zp - zd, ap - ad
        i1, i2 = ap - 2 * zp, ad - 2 * zd
        if dz == 0 and da == 0:
            return True, 0, 0, f"Identity (I={i1})"
        if dz < 0 or da < 0:
            return False, 0, 0, "Unreachable by a+EC: Z or A must increase"
        if da % 4:
            return False, 0, 0, f"Unreachable by a+EC: dA={da} not divisible by 4"
        k = da // 4
        c = dz - 2 * k
        if c < 0 or (ad - zd) < 0:
            return False, k, c, f"Unreachable by a+EC: I {i1}->{i2} (needs beta-)"
        return True, k, c, f"Reachable: alpha x{k} + EC x{c} (I {i1}->{i2})"

    @staticmethod
    def enumerate_nucleon_splits(dz: int, dn: int) -> Tuple[Tuple[int, int, int], ...]:
        if dz > 0 or dn > 0:
            return ()
        p, n = -dz, -dn
        return tuple((k, p - 2 * k, n - 2 * k) for k in range(min(p // 2, n // 2) + 1))

    @staticmethod
    def classify_event(zp, zd, tfi, atfi, flp, fld, blkp, blkd, mode="α") -> str:
        if mode != "α":
            return f"[{mode} EDIT] dZ={zd-zp:+d} (LSB {zp%2}->{zd%2}, wt={tfi})"
        for e in WPM.DIODE_CLOSURES:
            if (zp == e + 2 and zd == e) or (zp == e + 1 and zd == e - 1):
                kind = f"even barrier TFI={tfi}" if zp % 2 == 0 else "odd tunnel TFI=1"
                return f"[DIODE E={e}=6 mod 8] closure {e} ({kind})"
        if zd in (81, 82, 83):
            return f"[ATTRACTOR] classical bicone floor ({WPM.get_symbol(zd)} Z={zd})"
        if flp != fld:
            return f"[RIM / FLOOR WALL] S_{flp} -> S_{fld} (TFI={tfi})"
        if zp in (112, 113):
            return f"[SF WALL 6d10] closed-shell bottleneck (TFI={tfi})"
        if zp in (104, 105):
            return f"[SF WALL 6d/5f] d-exhaustion into f cavity (TFI={tfi})"
        if blkp == "f" and blkd == "f":
            return (f"[F-CAVITY SPIKE] borrow cascade, ATFI=0 (TFI={tfi})" if tfi >= 4
                    else f"[F-CAVITY] internal sqrt(5) resonance (ATFI=0, TFI={tfi})")
        if tfi >= 3:
            return f"[BORROW CASCADE] {tfi}-bit borrow wave"
        return f"[SMOOTH SLIDE] intra-{blkd} slide (TFI={tfi})"

    @classmethod
    def _make_step_record(cls, idx, mode, zc, ac, zn, an, unrelaxed_m) -> DecayStepRecord:
        before = ThreeRegisterState(zc, ac - zc, zc)
        unrel = ThreeRegisterState(zn, an - zn, unrelaxed_m)
        after = ThreeRegisterState(zn, an - zn, zn)
        mask, tfi = BinaryFrictionALU.compute_tfi(zc, zn)
        atfi, _ = BinaryFrictionALU.compute_atfi(zc, zn)
        flp, blkp, mup, _ = WPM.get_coordinates(zc)
        fld, blkd, mud, _ = WPM.get_coordinates(zn)
        tag = cls.classify_event(zc, zn, tfi, atfi, flp, fld, blkp, blkd, mode)
        hint = "-"
        if mode == "α" and zc >= 4:
            alt = BinaryFrictionALU.compute_tfi(zc - 1, zc - 3)[1]
            if zc % 4 == 0:
                hint = f"Z=0 mod 4: EC->{WPM.get_symbol(zc-1)}({zc-1}) lowers TFI {tfi}->{alt}"
            elif zc % 2 == 0:
                hint = f"Z=2 mod 4: EC->{WPM.get_symbol(zc-1)}({zc-1}) raises TFI {tfi}->{alt}"
        return DecayStepRecord(
            idx, mode, zc, ac, zn, an, format(zc, "08b"), format(zn, "08b"),
            mask, format(mask, "08b"), tfi, atfi,
            f"S{flp},{blkp},{mup}", f"S{fld},{blkd},{mud}", blkp, blkd, flp, fld,
            f"0x{WPM.nuclear_facing_word(zc):04X}",
            float(before.strain_proxy), float(after.strain_proxy),
            float(after.asymmetry_proxy), unrel.Q_top, tag, hint)

    @classmethod
    def trace_chain(cls, z_start, a_start, z_stop, ec_switches=None) -> ChainResult:
        if not (1 <= z_start <= 172):
            raise RegisterOverflow(f"start Z={z_start}")
        if a_start < z_start:
            raise PhysicalViolation(f"A={a_start} < Z={z_start} (N<0)")
        ec = set(ec_switches or ())
        z_stop = max(1, min(z_stop, z_start))
        recs: List[DecayStepRecord] = []
        zc, ac, step, halt = z_start, a_start, 1, ""
        while zc > z_stop and zc > 1:
            if zc in ec and zc - 1 >= z_stop:
                mode, zn, an, m = "EC/β⁺", zc - 1, ac, zc - 1
            else:
                if zc - 2 < 1:
                    halt = f"REGISTER FLOOR: Z={zc} cannot emit alpha (daughter Z<1)."
                    break
                if zc - 2 < z_stop:
                    halt = (f"PARITY RAIL: alpha from Z={zc} lands at {zc-2}, below "
                            f"Stop Z={z_stop}. Halted on-rail at Z={zc}. "
                            f"Use the EC switch to change parity.")
                    break
                if (ac - zc) < 2:
                    halt = (f"NEUTRON STARVATION: N={ac-zc}<2 at Z={zc}; "
                            f"alpha emission forbidden.")
                    break
                mode, zn, an, m = "α", zc - 2, ac - 4, zc
            recs.append(cls._make_step_record(step, mode, zc, ac, zn, an, m))
            zc, ac, step = zn, an, step + 1
        if not recs and not halt:
            halt = "No steps: start Z <= stop Z."
        return ChainResult(recs, halt, z_stop, zc, ac)

    @classmethod
    def canonical_path(cls, z1, a1, z2, a2) -> Tuple[List[DecayStepRecord], str]:
        """Provably physical ordering: EC first, then alpha; else alpha,p,n; else Z then N."""
        steps: List[DecayStepRecord] = []
        zc, ac, idx = z1, a1, 1

        def push(mode, zn, an, m):
            nonlocal zc, ac, idx
            steps.append(cls._make_step_record(idx, mode, zc, ac, zn, an, m))
            idx += 1
            zc, ac = zn, an

        ok, k, c, msg = cls.cascade_moves(z1, a1, z2, a2)
        dz, dn = z2 - z1, (a2 - z2) - (a1 - z1)
        if ok and (k or c):
            for _ in range(c):
                push("EC/β⁺", zc - 1, ac, zc - 1)
            for _ in range(k):
                push("α", zc - 2, ac - 4, zc)
            return steps, "EC-first / alpha-second: " + msg
        if dz < 0 and dn < 0:
            kk = min((-dz) // 2, (-dn) // 2)
            for _ in range(kk):
                push("α", zc - 2, ac - 4, zc)
            while zc > z2:
                push("1p-KO", zc - 1, ac - 1, zc)
            while ac > a2:
                push("1n-KO", zc, ac - 1, zc)
            return steps, f"knockout order alpha x{kk} -> p -> n (N monotone >= 0)"
        while zc > z2:
            push("1p-KO", zc - 1, ac - 1, zc)
        while zc < z2:
            push("+1p", zc + 1, ac + 1, zc)
        while ac > a2:
            push("1n-KO", zc, ac - 1, zc)
        while ac < a2:
            push("+1n", zc, ac + 1, zc)
        return steps, "generic order: Z-edit at fixed N, then N-edit at fixed Z"

    @staticmethod
    def validate_path(z1, a1, steps) -> Tuple[bool, str]:
        zc, ac = z1, a1
        for r in steps:
            if (r.z_parent, r.a_parent) != (zc, ac):
                return False, f"discontinuity at step {r.step_idx}"
            if not (1 <= r.z_daughter <= 172):
                return False, f"register overflow at step {r.step_idx}"
            if r.a_daughter < r.z_daughter:
                return False, f"N<0 at step {r.step_idx}"
            if r.mode == "α" and (ac - zc) < 2:
                return False, f"alpha with N<2 at step {r.step_idx}"
            zc, ac = r.z_daughter, r.a_daughter
        return True, "ok"

    @classmethod
    def solve_custom_transmutation(cls, z1, a1, z2, a2) -> CustomTransmutationResult:
        for z, a in ((z1, a1), (z2, a2)):
            if not (1 <= z <= 172):
                raise RegisterOverflow(f"Z={z} outside [1..172]")
            if a < z:
                raise PhysicalViolation(f"A={a} < Z={z}")
        n1, n2 = a1 - z1, a2 - z2
        dz, dn, da = z2 - z1, n2 - n1, a2 - a1
        xor_addr, tfi_addr = BinaryFrictionALU.compute_tfi(z1, z2)
        atfi_total, dblocks = BinaryFrictionALU.compute_atfi(z1, z2)
        atfi_sq, _ = BinaryFrictionALU.compute_atfi_sq(z1, z2)
        st1 = ThreeRegisterState(z1, n1, z1)
        st2u = ThreeRegisterState(z2, n2, z1)
        st2 = ThreeRegisterState(z2, n2, z2)
        mask_wt = wt(WPM.mask_int(z1) ^ WPM.mask_int(z2))
        terms = []
        if dz < 0:
            terms.append(f"{abs(dz)}p")
        elif dz > 0:
            terms.append(f"[-{dz}p capture]")
        if dn < 0:
            terms.append(f"{abs(dn)}n")
        elif dn > 0:
            terms.append(f"[-{dn}n capture]")
        rhs = " + ".join(terms) if terms else "0 (identity)"
        steps, note = cls.canonical_path(z1, a1, z2, a2)
        valid, vnote = cls.validate_path(z1, a1, steps)
        ok, k, c, reach = cls.cascade_moves(z1, a1, z2, a2)
        eq = (f"{a1}{WPM.get_symbol(z1)} (Z={z1},N={n1})  -->  "
              f"{a2}{WPM.get_symbol(z2)} (Z={z2},N={n2}) + {rhs}")
        return CustomTransmutationResult(
            z1, a1, n1, z2, a2, n2, dz, dn, da, xor_addr, tfi_addr, atfi_total, dblocks,
            BinaryFrictionALU.audit_nibble_borrow(z1, z2), st2u.Q_top, mask_wt,
            float(st1.strain_proxy), float(st2.strain_proxy),
            float(st1.asymmetry_proxy), float(st2.asymmetry_proxy), eq, steps,
            atfi_sq, cls.enumerate_nucleon_splits(dz, dn), ok, (k, c), reach,
            valid, f"{note} | validation: {vnote}")


# ============================================================================
# 5. HARDENING
# ============================================================================
GRAM_MINK = (8, 16, 24, 40, 48, 80, 120, 240)


@lru_cache(maxsize=8)
def minkowski_min_vectors(bound: int = 1) -> Tuple[Tuple[int, ...], ...]:
    """sigma(O_K): min ||.||^2 = 8 (AM-GM), equality only at +-1 (Kronecker)."""
    out = []
    for c in itertools.product(range(-bound, bound + 1), repeat=8):
        if any(c) and sum(g * ci * ci for g, ci in zip(GRAM_MINK, c)) == 8:
            out.append(c)
    return tuple(out)


def hermite_gamma_mink() -> float:
    return 8.0 / math.sqrt(240.0)


def fcc_fraction(N: int) -> float:
    """Lemma G: N cancels identically."""
    return (N * (4.0 / 3.0) * math.pi * (math.sqrt(2.0) / 4.0) ** 3) / (N / 4.0)


def digit_diff(a: int, b: int, base: int) -> int:
    da, db = [], []
    while a or b:
        da.append(a % base); db.append(b % base); a //= base; b //= base
    n = max(len(da), len(db))
    da += [0] * (n - len(da)); db += [0] * (n - len(db))
    return sum(x != y for x, y in zip(da, db))


def diode_scan(E: int = 86) -> Dict[int, Tuple[int, int]]:
    return {b: (digit_diff(E + 2, E, b), digit_diff(E + 1, E - 1, b)) for b in range(2, 17)}


FIREWALL_POOL = [1, 2, 3, 5, 6, 8, 10, 12, 13, 15, 16, 21, 24, 30, 32, 34, 43,
                 64, 85, 86, 103, 118, 136, 172, 240, 1728]


def simple_hits(t: int, pool=None) -> List[str]:
    P = [x for x in (pool or FIREWALL_POOL) if x != t]
    out = []
    for a in P:
        for b in P:
            for s, v in (("+", a + b), ("-", a - b), ("*", a * b), ("^", a ^ b), ("|", a | b)):
                if v == t:
                    out.append(f"{a}{s}{b}")
            if b and a % b == 0 and a // b == t:
                out.append(f"{a}/{b}")
        if a * a == t:
            out.append(f"{a}^2")
        if wt(a) == t:
            out.append(f"wt({a})")
        for s in range(1, 5):
            if (a << s) == t:
                out.append(f"{a}<<{s}")
    return out


def firewall_report(targets=(137, 139, 141, 143, 149, 151, 157)) -> Dict[int, int]:
    return {t: len(simple_hits(t)) for t in targets}


def grammar_coverage(lo=2, hi=256) -> float:
    return sum(1 for t in range(lo, hi + 1) if simple_hits(t)) / (hi - lo + 1)


def squarefree(n: int) -> int:
    x, i = abs(n), 2
    while i * i <= x:
        while x % (i * i) == 0:
            x //= i * i
        i += 1 if i == 2 else 2
    return x


def valley_rational(z: int) -> int:
    """Legacy 62/25 fit, kept only as a falsified null model."""
    q, r = divmod(62 * z + (50 if z % 2 else 0), 25)
    return q + 1 if r * 2 > 25 else q


ANTI_CLAIMS = [
    ("v17 9.3 'Z=120: alpha needs zero borrow'", lambda: wt(120 ^ 118) == 0),
    ("v17 9.3 'Z=126: catastrophic carry cascade'", lambda: wt(126 ^ 124) >= 4),
    ("v17 7.6 'bit 2^6 permanently 1 beyond equator'",
     lambda: all((z >> 6) & 1 for z in range(87, 173))),
    ("v17 5.2 'sqrt13 wall is a magnitude ceiling of K'",
     lambda: all(squarefree(d) not in {1, 2, 3, 5, 6, 10, 15, 30} for d in (15, 30))),
    ("E8 reading 'sigma(O_K) has kissing number 240'",
     lambda: len(minkowski_min_vectors()) == 240),
    ("FCC reading 'Kepler fraction singles out N=172'",
     lambda: abs(fcc_fraction(172) - fcc_fraction(164)) > 1e-12),
]


# ============================================================================
# 6. OVERLAP-PROOF RENDERING
# ============================================================================
EQ_GAP, CELL_W, CELL_H, GUTTER = 1.15, 0.90, 0.74, 9.0


def matrix_geometry():
    ymap = {fl: (11 - fl) - (EQ_GAP if fl >= 6 else 0.0) for fl in range(12)}
    return ymap, (-16.0 - GUTTER - 1.0, 17.0), (ymap[11] - 1.0, ymap[0] + 1.7)


def cell_xy(fl, mu, cap, ymap):
    return (mu - 1) - cap / 2.0 + 0.5, ymap[fl]


def axes_pt_scale(fig, rect, xlim, ylim):
    fw, fh = fig.get_size_inches()
    return (fw * rect[2] * 72.0 / (xlim[1] - xlim[0]),
            fh * rect[3] * 72.0 / (ylim[1] - ylim[0]))


def fit_fs(text, cw_pt, ch_pt, cap, fw=0.86, fh=0.46):
    if not text:
        return 2.0
    return max(2.0, min(cap, fw * cw_pt / (0.62 * len(text)), fh * ch_pt))


def _draw_pixel_grid(fig, rect, color_fn, alpha_fn, edge_fn, ls_fn):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.axis("off")
    ymap, xlim, ylim = matrix_geometry()
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    px, py = axes_pt_scale(fig, rect, xlim, ylim)
    cw, ch = CELL_W * px, CELL_H * py
    for z in range(1, 173):
        fl, blk, mu, _ = WPM.get_coordinates(z)
        xc, yc = cell_xy(fl, mu, WPM.CAPACITIES[fl], ymap)
        ax.add_patch(Rectangle((xc - CELL_W / 2, yc - CELL_H / 2), CELL_W, CELL_H,
                               facecolor=color_fn(z, blk), alpha=alpha_fn(z),
                               edgecolor=edge_fn(z, blk), lw=0.9, ls=ls_fn(z)))
        sym, num = WPM.cell_labels(z)
        f1 = fit_fs(sym, cw, ch, 8.0, 0.86, 0.46)
        f2 = fit_fs(num, cw, ch, 6.4, 0.86, 0.32) if num else 0.0
        light = "#FFFFFF" if alpha_fn(z) > 0.3 else "#7E8CA6"
        if not num or f1 < 3.4:
            ax.text(xc, yc, sym, ha="center", va="center",
                    fontsize=f1, fontweight="bold", color=light)
        else:
            ax.text(xc, yc + 0.115, sym, ha="center", va="center",
                    fontsize=f1, fontweight="bold", color=light)
            ax.text(xc, yc - 0.150, num, ha="center", va="center",
                    fontsize=f2, color=light)
    for fl in range(12):
        lab = f"S{fl}   C={WPM.CAPACITIES[fl]}   Sum={WPM.CLOSURES[fl]}"
        f = max(3.0, min(8.5, 0.92 * GUTTER * px / (0.58 * len(lab))))
        ax.text(-16.8, ymap[fl], lab, ha="right", va="center",
                fontsize=f, fontweight="bold", color=CYAN)
    return ax, ymap, px, py, cw, ch


def render_matrix(fig, records=None, idx=0):
    fig.clear()
    fig.patch.set_facecolor(BG)
    ax, ymap, px, py, cw, ch = _draw_pixel_grid(
        fig, [0.012, 0.02, 0.976, 0.905],
        lambda z, b: BLOCK_COLORS[b],
        lambda z: 0.22 if z >= 119 else 0.42,
        lambda z, b: BLOCK_COLORS[b],
        lambda z: "--" if z >= 119 else "-")
    eq = (ymap[5] + ymap[6]) / 2.0
    txt = ("RELATIVISTIC EQUATOR  Z=86.5   |   edge 86|87 = 0x56|0x57   |   "
           "sum 173 = 0xAD, low nibble D = 13 (Kummer wall)")
    f_eq = max(3.0, min(8.6, 0.50 * EQ_GAP * py, 0.95 * (33.0 * px) / (0.58 * len(txt))))
    ax.plot([-16, 16], [eq, eq], color=PINK, lw=1.3, ls="--", zorder=1)
    ax.text(0, eq, txt, ha="center", va="center", fontsize=f_eq, color=PINK,
            fontweight="bold", zorder=3,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=BG, edgecolor=PINK, alpha=0.95))
    if records:
        chain = [records[0].z_parent] + [r.z_daughter for r in records]
        pts = [cell_xy(*WPM.get_coordinates(z)[0:1],
                       WPM.get_coordinates(z)[2], WPM.CAPACITIES[WPM.get_coordinates(z)[0]],
                       ymap) for z in chain]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GREEN, lw=2.2,
                marker="o", markersize=5.0, zorder=10,
                label=f"trajectory Z={chain[0]} -> {chain[-1]}")
        r = records[max(0, min(idx, len(records) - 1))]
        for z in (r.z_parent, r.z_daughter):
            fl, _, mu, _ = WPM.get_coordinates(z)
            x, y = cell_xy(fl, mu, WPM.CAPACITIES[fl], ymap)
            ax.scatter([x], [y], color=AMBER, s=210, zorder=12, edgecolors=BG, lw=1.8)
    for b, c in BLOCK_COLORS.items():
        ax.scatter([], [], color=c, s=80, label=f"{b}-block")
    ax.scatter([], [], facecolor="none", edgecolor=TXT, ls="--", s=80, label="predicted Z>=119")
    ax.legend(loc="lower right", ncol=6, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TXT, fontsize=8.0)
    ax.set_title("IT3 TOPOLOGICAL PERIODIC MATRIX  ::  172-layer wave-pixel bicone"
                 "   (Victor Logvinovich & Jean-Claude Perez)",
                 fontsize=11.0, fontweight="bold", color=CYAN, pad=6)


def render_cern(fig, res):
    fig.clear()
    fig.patch.set_facecolor(BG)
    if res is None:
        return
    fig.suptitle(f"TRANSMUTATION LAB :: {res.direct_equation}",
                 fontsize=11.0, fontweight="bold", color=AMBER)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45], hspace=0.30, wspace=0.20,
                          left=0.06, right=0.97, top=0.90, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0], facecolor=PANEL)
    mat = np.array([[(res.z1 >> (7 - b)) & 1 for b in range(8)],
                    [(res.z2 >> (7 - b)) & 1 for b in range(8)],
                    [(res.xor_addr >> (7 - b)) & 1 for b in range(8)]])
    ax.imshow(mat, aspect="auto", cmap="plasma", vmin=0, vmax=1.25)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f"Z1={res.z1} (0x{res.z1:02X})", f"Z2={res.z2} (0x{res.z2:02X})",
                        f"XOR 0x{res.xor_addr:02X} wt={res.tfi_addr}"],
                       fontsize=8.2, fontweight="bold", color=TXT)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"b{7-b}" for b in range(8)], fontsize=8, color=TXT)
    ax.axvline(3.5, color=CYAN, lw=1.8, ls="--")
    for i in range(3):
        for j in range(8):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=BG if mat[i, j] else "#8A99AD")
    ax.set_title(f"Address ALU & nibble borrow (ATFI={res.atfi_total:.2f})",
                 fontsize=9.4, fontweight="bold", color=CYAN)
    ax.tick_params(colors=TXT)

    ax2 = fig.add_subplot(gs[0, 1], facecolor=PANEL)
    if res.stepwise_records:
        zs = [res.z1] + [r.z_daughter for r in res.stepwise_records]
        ns = [res.n1] + [r.a_daughter - r.z_daughter for r in res.stepwise_records]
        ax2.plot(ns, zs, color=CYAN, lw=2.0, marker="o", markersize=5.0, label="canonical path")
    ax2.scatter([res.n1], [res.z1], color=AMBER, s=130, zorder=5, edgecolors="white",
                label=f"{res.a1}{WPM.get_symbol(res.z1)}")
    ax2.scatter([res.n2], [res.z2], color=PINK, s=150, marker="*", zorder=6,
                edgecolors="white", label=f"{res.a2}{WPM.get_symbol(res.z2)}")
    ax2.set_title(f"(Z,N) trajectory | dZ={res.dz:+d} dN={res.dn:+d} | "
                  f"path {'VALID' if res.path_valid else 'INVALID'}",
                  fontsize=9.4, fontweight="bold", color=GREEN)
    ax2.set_xlabel("N = A - Z", fontsize=8.5, color=TXT)
    ax2.set_ylabel("Z", fontsize=8.5, color=TXT)
    ax2.grid(True, color=GRID, ls="--", alpha=0.5)
    ax2.tick_params(colors=TXT, labelsize=8)
    ax2.legend(loc="best", facecolor=BG, edgecolor=GRID, labelcolor=TXT, fontsize=7.6)

    zmin, zmax = min(res.z1, res.z2), max(res.z1, res.z2)
    axm, ymap, px, py, cw, ch = _draw_pixel_grid(
        fig, [0.045, 0.045, 0.94, 0.44],
        lambda z, b: CYAN if z <= zmin else (PINK if z <= zmax else "#182034"),
        lambda z: 0.45 if z <= zmin else (0.92 if z <= zmax else 0.35),
        lambda z, b: "#FFFFFF" if zmin < z <= zmax else "#26324F",
        lambda z: "-")
    axm.scatter([], [], color=CYAN, s=80, label=f"common core 1..{zmin}")
    axm.scatter([], [], color=PINK, s=90, label=f"mask XOR diff ({res.mask_xor_weight} bits)")
    axm.scatter([], [], color="#182034", s=80, label=f"empty {zmax+1}..172")
    axm.legend(loc="lower right", ncol=3, facecolor=BG, edgecolor=GRID,
               labelcolor=TXT, fontsize=8.0)
    axm.set_title(f"172-bit mask audit: wt(M_{res.z1} XOR M_{res.z2}) = "
                  f"{res.mask_xor_weight} | immediate Q_top = {res.q_top_unrelaxed:+d}",
                  fontsize=9.6, fontweight="bold", color=AMBER, pad=4)


def render_hud(fig, records, idx=0):
    fig.clear()
    fig.patch.set_facecolor(BG)
    if not records:
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.5, 0.5, "no cascade records", ha="center", color=TXT)
        return
    fw, fh = fig.get_size_inches()
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 0.9], width_ratios=[1.3, 1.0],
                          hspace=0.50, wspace=0.22, left=0.08, right=0.97,
                          top=0.91, bottom=0.07)
    zd = [r.z_daughter for r in records]
    zp = [r.z_parent for r in records]
    tfis = [r.tfi for r in records]
    atfis = [r.atfi for r in records]
    strains = [r.strain_after for r in records]
    act = records[max(0, min(idx, len(records) - 1))]
    n = len(records)
    span_pt = fw * 0.89 * 72.0
    step_pt = span_pt / max(1, n)

    ax1 = fig.add_subplot(gs[0, :], facecolor=PANEL)
    ax1.plot(zd, tfis, color=CYAN, lw=2.1, marker="o", markersize=5.0,
             label="TFI = wt(Z XOR Z')")
    ax1.plot(zd, atfis, color=AMBER, lw=1.6, ls="--", marker="s", markersize=4.0,
             label="ATFI on T3(1,sqrt2,sqrt3)")
    ax1.axhline(2.5, color="#FFFFFF", ls=":", lw=1.0, alpha=0.6, label="wall threshold TFI>=3")
    ax1.axvline(act.z_daughter, color=GREEN, ls="-.", lw=1.5,
                label=f"step {act.step_idx}: {WPM.get_symbol(act.z_parent)}->"
                      f"{WPM.get_symbol(act.z_daughter)}")
    walls = [r for r in records if r.tfi >= 3]
    ax1.scatter([r.z_daughter for r in walls], [r.tfi for r in walls],
                color=PINK, s=110, zorder=6, edgecolors="white", lw=1.1)
    keep_every = max(1, int(math.ceil(52.0 / max(step_pt, 1e-6))))
    shown = 0
    for i, r in enumerate(walls):
        if shown >= 10:
            break
        if (i % keep_every) and (r is not act):
            continue
        ax1.annotate(f"{WPM.get_symbol(r.z_parent)}->{WPM.get_symbol(r.z_daughter)} TFI={r.tfi}",
                     (r.z_daughter, r.tfi), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=7.2, fontweight="bold", color=PINK)
        shown += 1
    rail = "even rail LSB=0" if zp[0] % 2 == 0 else "odd rail LSB=1"
    ax1.set_title(f"ALPHA-CASCADE ENGINE :: {WPM.get_symbol(zp[0])}(Z={zp[0]}) -> "
                  f"{WPM.get_symbol(zd[-1])}(Z={zd[-1]}) [{rail}]",
                  fontsize=10.2, fontweight="bold", color=CYAN, pad=8)
    ax1.set_ylabel("friction index", color=TXT, fontsize=9)
    ax1.set_ylim(-0.4, max(max(tfis), 4) + 2.8)
    ax1.invert_xaxis()
    ax1.grid(True, color=GRID, ls="--", alpha=0.5)
    ax1.tick_params(colors=TXT, labelsize=8.2)
    ax1.legend(loc="upper center", ncol=4, facecolor=BG, edgecolor=GRID,
               labelcolor=TXT, fontsize=7.4)

    ax2 = fig.add_subplot(gs[1, 0], facecolor=PANEL)
    allz = [zp[0]] + zd
    bits = np.zeros((8, len(allz)), dtype=int)
    for c, z in enumerate(allz):
        for b in range(8):
            bits[7 - b, c] = (z >> b) & 1
    ax2.imshow(bits, aspect="auto", cmap="magma", interpolation="nearest", vmin=0, vmax=1.2)
    ax2.set_yticks(range(8))
    ax2.set_yticklabels([f"b{7-i} ({2**(7-i)})" for i in range(8)], fontsize=7.2, color=TXT)
    half_pt = fw * 0.40 * 72.0
    tick_every = max(1, int(math.ceil(20.0 / max(half_pt / max(1, len(allz)), 1e-6))))
    ticks = list(range(0, len(allz), tick_every))
    if (len(allz) - 1) not in ticks:
        ticks.append(len(allz) - 1)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([str(allz[i]) for i in ticks], fontsize=7.0, rotation=45, color=TXT)
    if len(allz) <= 26:
        for i in range(8):
            for c in range(len(allz)):
                ax2.text(c, i, str(bits[i, c]), ha="center", va="center", fontsize=6.4,
                         fontweight="bold", color=BG if bits[i, c] else "#6C7A9C")
    ax2.axvline(idx + 1, color=GREEN, lw=1.8)
    ax2.set_title("8-bit address register trace", fontsize=9.6, fontweight="bold", color=TXT)
    ax2.tick_params(colors=TXT)

    ax3 = fig.add_subplot(gs[1, 1], facecolor=PANEL)
    ax3.plot(zd, strains, color=GREEN, lw=2.0, marker="^", markersize=4.5, label="strain = Z^2/A")
    ax3.axvline(act.z_daughter, color=PINK, ls="-.", lw=1.3)
    ax3.invert_xaxis()
    ax3.grid(True, color=GRID, ls="--", alpha=0.5)
    ax3.tick_params(colors=TXT, labelsize=8.2)
    ax3.set_title("Topological strain relaxation", fontsize=9.6, fontweight="bold", color=TXT)
    ax3.set_xlabel("daughter Z", fontsize=8.4, color=TXT)
    ax3.legend(loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=TXT, fontsize=7.6)

    ax4 = fig.add_subplot(gs[2, :], facecolor=PANEL)
    ax4.axhline(0, color=GRID, lw=4, zorder=1)
    lab_every = max(1, int(math.ceil(30.0 / max(step_pt, 1e-6))))
    for i, r in enumerate(records):
        cur = (i == idx)
        ax4.scatter([r.z_daughter], [0], color=BLOCK_COLORS[r.block_daughter],
                    s=320 if cur else 180, zorder=4,
                    edgecolors=GREEN if cur else "white", lw=2.2 if cur else 1.0)
        if cur or i % lab_every == 0 or i == n - 1:
            ax4.text(r.z_daughter, 0.24, f"{WPM.get_symbol(r.z_daughter)} {r.z_daughter}",
                     ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=TXT)
            ax4.text(r.z_daughter, -0.26, f"S{r.floor_daughter},{r.block_daughter}",
                     ha="center", va="top", fontsize=6.2, color="#9AA5B1")
    for b, c in BLOCK_COLORS.items():
        ax4.scatter([], [], color=c, s=80, label=f"{b}-block")
    ax4.set_ylim(-0.7, 0.7)
    ax4.set_yticks([])
    ax4.invert_xaxis()
    ax4.tick_params(colors=TXT, labelsize=8.2)
    ax4.set_xlabel("atomic number Z (labels decimated to prevent overlap)", fontsize=8.6, color=TXT)
    ax4.set_title("Oh subshell branching track", fontsize=9.6, fontweight="bold", color=TXT)
    ax4.legend(loc="upper right", ncol=4, facecolor=BG, edgecolor=GRID,
               labelcolor=TXT, fontsize=7.6)


# ============================================================================
# 7. SELF-TEST
# ============================================================================
def _subtractive_row(n: int) -> List[int]:
    row = [1]
    for m in range(1, n + 1):
        nxt = [1] * (m + 1)
        for k in range(1, m):
            nxt[k] = row[k - 1] - (row[k] if k < len(row) else 0)
        row = nxt
    return row


def build_ledger() -> Ledger:
    L = Ledger()
    E = AlphaCascadeEngine

    qs = [(min(n, 11 - n) + 3) // 2 for n in range(12)]
    L.check("A", "A01 capacity law, sum q^2 = 43/86, total 172",
            lambda: qs == [1, 2, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1]
            and sum(q * q for q in qs[:6]) == 43 and sum(q * q for q in qs) == 86
            and WPM.CAPACITIES == (2, 8, 8, 18, 18, 32, 32, 18, 18, 8, 8, 2)
            and sum(WPM.CAPACITIES) == 172 == (43 << 2) == (86 << 1))

    def facing():
        cs = set(WPM.CLOSURES)
        bc = {"s": 0, "p": 0, "d": 0, "f": 0}
        for z in range(1, 173):
            fl, blk, mu, _ = WPM.get_coordinates(z)
            bc[blk] += 1
            fz = WPM.facing_partner(z)
            f2, b2, m2, _ = WPM.get_coordinates(fz)
            if not (1 <= fz <= 172 and fz != z and WPM.facing_partner(fz) == z):
                return False
            if fz % 2 != z % 2 or f2 != 11 - fl or b2 != blk or m2 != mu:
                return False
            if WPM.closure_distance(fz) != WPM.closure_distance(z):
                return False
            w, wf = WPM.nuclear_facing_word(z), WPM.nuclear_facing_word(fz)
            if wf != (((w & 0xFF) << 8) | ((w >> 8) & 0xFF)):
                return False
        if not all(WPM.facing_partner(s) in cs for s in WPM.CLOSURES):
            return False
        return (bc == {"s": 24, "p": 60, "d": 60, "f": 28}
                and WPM.facing_partner(86) == 118 and WPM.facing_partner(54) == 136)
    L.check("A", "A02 facing involution, floor reflection, blocks (24,60,60,28)", facing)

    def tfi_core():
        for m in range(1, 2049):
            if wt(m ^ (m - 1)) != nu2(m) + 1:
                return False
        for z in range(2, 2049):
            exp = nu2(z) if z % 2 == 0 else nu2(z - 1)
            if wt(z ^ (z - 2)) != exp:
                return False
        if WPM.DIODE_CLOSURES != (54, 86, 118):
            return False
        for e in WPM.DIODE_CLOSURES:
            if wt((e + 2) ^ e) < 3 or wt((e + 1) ^ (e - 1)) != 1:
                return False
        d = r = t = 0
        for z in range(4, 173):
            a, b = wt(z ^ (z - 2)), wt((z - 1) ^ (z - 3))
            if z % 2:
                if a != b:
                    return False
                t += 1
            elif z % 4 == 0:
                if a <= b:
                    return False
                d += 1
            else:
                if a >= b:
                    return False
                r += 1
        return (d, r, t) == (43, 42, 84)
    L.check("A", "A03 borrow lemma, 2-adic TFI, diodes, EC detour 43/42/84", tfi_core)

    def atfi():
        cases = {(4, 2): 4, (21, 19): 4, (10, 8): 8, (30, 28): 12,
                 (96, 94): 0, (88, 86): 4, (82, 79): 11}
        for (a, b), v in cases.items():
            if BinaryFrictionALU.compute_atfi_sq(a, b)[0] != v:
                return False
        w = {"s": 1, "p": 2, "d": 3, "f": 0}
        hist: Dict[int, int] = {}
        for z in range(2, 173):
            bh = WPM.get_coordinates(z)[1]
            bl = WPM.get_coordinates(z - 1)[1]
            sq = 4 * w[bh] if bh == bl else w[bh] + w[bl]
            if sq != BinaryFrictionALU.compute_atfi_sq(z, z - 2)[0]:
                return False
            hist[sq] = hist.get(sq, 0) + 1
        return hist == {0: 26, 1: 2, 3: 16, 4: 17, 5: 6, 8: 50, 12: 54}
    L.check("A", "A04 integer ATFI^2 and single-alpha histogram", atfi)

    def masks_strain():
        for a in range(0, 173, 7):
            for b in range(0, 173, 5):
                if wt(WPM.mask_int(a) ^ WPM.mask_int(b)) != abs(a - b):
                    return False
        for z in range(3, 81):
            for nn in range(2, 61):
                a = z + nn
                if (z - 2) ** 2 * a >= z * z * (a - 4):
                    return False
            if not (ThreeRegisterState(z, 63, z).strain_proxy
                    < ThreeRegisterState(z, 60, z).strain_proxy):
                return False
        return True
    L.check("A", "A05 prefix-mask XOR metric and alpha strain lemma", masks_strain)

    def nibble():
        for b in (2, 3, 4, 8, 16):
            for N in list(range(1, 26)) + [172, 255]:
                bo, hi, qu, fp = nibble_brute(N, b)
                if (bo, hi, qu) != nibble_closed(N, b):
                    return False
                if bo + hi + qu != N * (N + 1) // 2:
                    return False
        return nibble_brute(172, 16) == (6570, 7030, 1278, 6873)
    L.check("A", "A06 Thm E nibble closed forms (any N/base) + 172 witness", nibble)

    L.check("A", "A07 legacy 62/25 valley fit hits only 4 references",
            lambda: [z for z, m in WPM.STANDARD_MASS.items() if valley_rational(z) == m]
            == [70, 74, 119, 120] and valley_rational(172) == 427)

    def algebra():
        adm = [d for d in range(1, 31) if 30 % d == 0 and squarefree(d) == d]
        if adm != [1, 2, 3, 5, 6, 10, 15, 30]:
            return False
        S = set(adm)
        if [squarefree(x) in S for x in (1, 2, 3, 5, 8)] != [True] * 5:
            return False
        if squarefree(13) in S:
            return False
        code = {1: 0, 2: 1, 3: 2, 5: 4, 6: 3, 10: 5, 15: 6, 30: 7}
        for a in code:
            for b in code:
                if (code[a] ^ code[b]) != code[squarefree(a * b)]:
                    return False
        prod = 8 * 12 * 5 * 24 * 40 * 60 * 120
        if prod != 2 ** 16 * 3 ** 4 * 5 ** 4 or prod != 3_317_760_000:
            return False
        if prod != 57_600 ** 2:
            return False
        if [i for i in range(16) if (0xE100 >> i) & 1] != [8, 13, 14, 15]:
            return False
        for nn in range(1, 33):
            row = _subtractive_row(nn)
            if sum(row) != 2:
                return False
            for k, v in enumerate(row):
                if (v % 2) != (0 if (k & ~nn) else 1):
                    return False
        if (2 ** 6) // 3 != 21 or (2 ** 8) // 3 != 85 or 2 * 21 + 1 != 43:
            return False
        if not all(p % 24 == 19 for p in (19, 43, 67, 163)):
            return False
        return 13 * 13 - 32 == 137 == 240 - 103 == (136 | 1) == 0x89 and wt(137) == 3
    L.check("A", "A08 squareclass XOR, discriminant, Pascal/Lucas, Heegner", algebra)

    def reach():
        checked = 0
        for z in range(1, 31):
            for nn in range(0, 13):
                a = z + nn
                seen, stack = set(), [(z, a)]
                while stack:
                    zz, aa = stack.pop()
                    if (zz, aa) in seen:
                        continue
                    seen.add((zz, aa))
                    if zz >= 3 and aa - 4 >= zz - 2:
                        stack.append((zz - 2, aa - 4))
                    if zz >= 2:
                        stack.append((zz - 1, aa))
                for z2 in range(1, z + 1):
                    for n2 in range(0, nn + (z - z2) + 1):
                        a2 = z2 + n2
                        if E.cascade_moves(z, a, z2, a2)[0] is not ((z2, a2) in seen):
                            return False
                        checked += 1
        return checked == 100750
    L.check("A", "A09 alpha+EC reachability vs DFS (100750 pairs)", reach)

    def paths():
        gz = [1, 5, 17, 43, 82, 118, 172]
        for z1 in gz:
            for n1 in (0, 3, 8):
                for z2 in gz:
                    for n2 in (0, 4, 10):
                        a1, a2 = z1 + n1, z2 + n2
                        steps, _ = E.canonical_path(z1, a1, z2, a2)
                        ok, why = E.validate_path(z1, a1, steps)
                        if not ok:
                            raise AssertionError(f"{z1},{a1}->{z2},{a2}: {why}")
                        if steps and (steps[-1].z_daughter, steps[-1].a_daughter) != (z2, a2):
                            return False
                        if not steps and (z1, a1) != (z2, a2):
                            return False
        return True
    L.check("A", "A10 canonical_path physical and exact on a fixed grid", paths)

    def solver():
        r = E.solve_custom_transmutation(82, 208, 79, 197)
        if r.nucleon_splits != ((0, 3, 8), (1, 1, 6)) or r.alpha_ec_reachable:
            return False
        if not r.path_valid:
            return False
        checks = [((92, 238, 82, 206), (False, 8, -6)), ((92, 238, 82, 218), (True, 5, 0)),
                  ((118, 294, 82, 222), (True, 18, 0)), ((117, 293, 83, 225), (True, 17, 0)),
                  ((112, 282, 82, 222), (True, 15, 0))]
        for args, exp in checks:
            if E.cascade_moves(*args)[:3] != exp:
                return False
        if WPM.parse_nuclide_input("208Pb") != (82, 208):
            return False
        if WPM.parse_nuclide_input("Au-197") != (79, 197):
            return False
        if WPM.parse_nuclide_input("82,208") != (82, 208):
            return False
        for bad in ("Pb", "82", "Lead", "E172", "82,10", "26,12"):
            try:
                WPM.parse_nuclide_input(bad)
                return False
            except ValueError:
                pass
        return True
    L.check("A", "A11 solver splits, preset endpoints, strict parser", solver)

    L.check("A", "A12 Thm D: TFI_{2^j}(Z)=nu2(Z>>j)+1, j<=8, Z<=2048",
            lambda: all(BinaryFrictionALU.tfi_pow2(z, j) == nu2(z >> j) + 1
                        for j in range(9) for z in range(1 << j, 2049)))
    L.check("A", "A13 Thm F: Minkowski min 8, kissing number = 2",
            lambda: len(minkowski_min_vectors()) == 2)
    L.check("A", "A14 Thm F': Hermite gamma 0.516 != 2 (not isometric to E8)",
            lambda: abs(hermite_gamma_mink() - 2.0) > 1.0)
    L.check("A", "A15 Lemma G: FCC packing fraction independent of N",
            lambda: all(abs(fcc_fraction(N) - math.pi / math.sqrt(18)) < 1e-12
                        for N in range(4, 2001, 4)))
    L.check("A", "A16 base scan: diode in base 2, absent in base 3",
            lambda: diode_scan()[2] == (3, 1) and diode_scan()[3] == (2, 2))

    def guards():
        try:
            WPM.get_coordinates(173)
            return False
        except RegisterOverflow:
            pass
        try:
            ThreeRegisterState(5, -1, 5)
            return False
        except PhysicalViolation:
            pass
        ch = E.trace_chain(20, 20, 2)
        if ch.records or "NEUTRON STARVATION" not in ch.halt:
            return False
        ch2 = E.trace_chain(120, 298, 81)
        if "PARITY RAIL" not in ch2.halt:
            return False
        if ch2.records and ch2.records[-1].z_daughter < 81:
            return False
        return True
    L.check("A", "A17 guards: overflow, N<0, N<2 alpha, parity rail", guards)

    for name, fn in ANTI_CLAIMS:
        L.check("REFUTED", name, fn, expect=False, note="must stay false (regression)")
    return L


def print_diagnostics() -> None:
    print("\n  --- DIAGNOSTICS (arithmetic facts; interpretation is Level O) ---")
    print("  base scan (even,odd):",
          "  ".join(f"b{b}:{v[0]}/{v[1]}" for b, v in diode_scan().items()))
    print("  -> diode maximal in base 2, vanishes in bases 3,5,16.")
    fw = firewall_report()
    print("\n  --- NUMEROLOGY FIREWALL (<=2 ops, target excluded from pool) ---")
    print("  hits:", "  ".join(f"{k}:{v}" for k, v in fw.items()))
    print(f"  grammar coverage [2,256] = {grammar_coverage():.3f}")
    ctrl = [v for k, v in fw.items() if k != 137]
    if ctrl and fw[137] <= max(ctrl):
        print("  VERDICT: 137 NOT distinguished from sham targets -> Level C only.")
    else:
        print("  VERDICT: 137 exceeds controls; still needs an out-of-sample test.")
    print("\n  --- QUARANTINE (B/C/O) ---")
    print("   137 as QED alpha; 57600=240^2 as physical E8xE8; Ry/3 as the H2 curve;")
    print("   greedy cascade as a CERN mechanism; Z=172 as established fact;")
    print("   TFI as nuclear dynamics (confounded with Z-parity / pairing).")


def run_self_test(json_path: Optional[str] = None) -> int:
    print("=" * 92)
    print(f" IT3-WavePixel v{VERSION} :: HARDENED LEVEL-A LEDGER")
    print(" Victor Logvinovich & Jean-Claude Perez")
    print("=" * 92)
    L = build_ledger()
    L.report()
    print_diagnostics()
    if json_path:
        L.to_json(json_path)
        print(f"\n  ledger JSON: {json_path}")
    print("=" * 92)
    return 0 if L.n_pass == L.n else 1


# ============================================================================
# 8. CLI
# ============================================================================
def cli_transmute(src: str, dst: str) -> int:
    z1, a1 = WPM.parse_nuclide_input(src)
    z2, a2 = WPM.parse_nuclide_input(dst)
    r = AlphaCascadeEngine.solve_custom_transmutation(z1, a1, z2, a2)
    lbl = "2-adic TFI" if (z1 - z2 == 2) else "address Hamming wt"
    print("=" * 100)
    print(f" {r.direct_equation}")
    print("=" * 100)
    print(f"  source : Z={r.z1} N={r.n1} I={r.a1-2*r.z1}  0x{r.z1:02X} {r.z1:08b}")
    print(f"  target : Z={r.z2} N={r.n2} I={r.a2-2*r.z2}  0x{r.z2:02X} {r.z2:08b}")
    print(f"  edit   : dZ={r.dz:+d} dN={r.dn:+d} dA={r.da:+d}")
    print(f"  ALU    : XOR=0x{r.xor_addr:02X} {lbl}={r.tfi_addr} | ATFI^2={r.atfi_sq} "
          f"(ATFI={r.atfi_total:.3f})")
    print(f"  nibble : {r.nibble_report}")
    print("  splits : " + ("; ".join(f"k={k}: a x{k} + p x{p} + n x{n}"
                                     for k, p, n in r.nucleon_splits) or "N/A"))
    print(f"  a+EC   : {r.reachability_note}")
    print(f"  mask   : Q_top={r.q_top_unrelaxed:+d} | wt(M^M')={r.mask_xor_weight}")
    print(f"  path   : {'VALID' if r.path_valid else 'INVALID'} :: {r.path_note}")
    print("-" * 100)
    for s in r.stepwise_records:
        print(f"   {s.step_idx:>2} [{s.mode:<7}] {s.a_parent}{WPM.get_symbol(s.z_parent):<4}"
              f" -> {s.a_daughter}{WPM.get_symbol(s.z_daughter):<4} | {s.bin_parent}->"
              f"{s.bin_daughter} | wt={s.tfi} | ATFI={s.atfi:.2f} | {s.event_tag}")
    print("=" * 100)
    return 0


def batch_export(outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    for z, a, stop in [(120, 298, 82), (119, 297, 81), (118, 294, 82),
                       (112, 282, 82), (172, 426, 82), (92, 238, 82)]:
        ch = AlphaCascadeEngine.trace_chain(z, a, stop)
        if not ch.records:
            continue
        for name, fn in (("hud", lambda f: render_hud(f, ch.records, 0)),
                         ("matrix", lambda f: render_matrix(f, ch.records, 0))):
            fig = Figure(figsize=(15, 9.5), dpi=110, facecolor=BG)
            fn(fig)
            fig.savefig(os.path.join(outdir, f"IT3_{name}_Z{z}.png"), dpi=150, facecolor=BG)
        with open(os.path.join(outdir, f"IT3_audit_Z{z}.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "mode", "Zp", "Ap", "Zd", "Ad", "XOR", "TFI", "ATFI",
                        "coord_d", "W_N", "strain_d", "dQ_top", "event", "hint"])
            for r in ch.records:
                w.writerow([r.step_idx, r.mode, r.z_parent, r.a_parent, r.z_daughter,
                            r.a_daughter, f"0x{r.xor_mask:02X}", r.tfi, f"{r.atfi:.4f}",
                            r.coord_daughter, r.facing_word_hex, f"{r.strain_after:.4f}",
                            r.q_top_unrelaxed, r.event_tag, r.tunnel_hint])
    res = AlphaCascadeEngine.solve_custom_transmutation(82, 208, 79, 197)
    fig = Figure(figsize=(15, 10), dpi=110, facecolor=BG)
    render_cern(fig, res)
    fig.savefig(os.path.join(outdir, "IT3_cern_Pb_Au.png"), dpi=150, facecolor=BG)
    build_ledger().to_json(os.path.join(outdir, "IT3_ledger.json"))
    print(f"batch export -> {os.path.abspath(outdir)}")
    return 0


# ============================================================================
# 9. GUI
# ============================================================================
if HAS_PYQT:
    class IT3Suite(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"IT3-WavePixel v{VERSION} :: Hardened Cascade & "
                                f"Transmutation Suite [Logvinovich & Perez]")
            self.resize(1720, 1010)
            self.records: List[DecayStepRecord] = []
            self.halt_msg = ""
            self.ec_switches: set = set()
            self.idx = 0
            self.last: Optional[CustomTransmutationResult] = None
            self.timer = QTimer(self)
            self.timer.setInterval(650)
            self.timer.timeout.connect(self._advance)
            self._style()
            self._build()
            self.recompute()
            self.transmute()

        def _style(self):
            self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{BG};color:{TXT};
                font-family:'Menlo','Consolas',monospace;font-size:12px;}}
            QGroupBox{{background:#111625;border:1px solid {GRID};border-radius:6px;
                margin-top:10px;padding-top:10px;font-weight:bold;color:{CYAN};}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 6px;}}
            QPushButton{{background:#19223A;color:{CYAN};border:1px solid {CYAN};
                border-radius:4px;padding:5px 9px;font-weight:bold;}}
            QPushButton:hover{{background:{CYAN};color:{BG};}}
            QPushButton#gold{{background:#2B2108;color:{AMBER};border:1px solid {AMBER};}}
            QPushButton#gold:hover{{background:{AMBER};color:{BG};}}
            QPushButton#pink{{background:#2A1228;color:{PINK};border:1px solid {PINK};}}
            QPushButton#pink:hover{{background:{PINK};color:{BG};}}
            QSpinBox,QComboBox,QLineEdit{{background:#0D121F;color:{AMBER};
                border:1px solid #2E3D61;border-radius:4px;padding:4px;font-weight:bold;}}
            QTabWidget::pane{{border:1px solid {GRID};background:#0E1320;}}
            QTabBar::tab{{background:#13192B;color:#9AA5B1;border:1px solid {GRID};
                padding:8px 14px;font-weight:bold;}}
            QTabBar::tab:selected{{background:#1B2642;color:{CYAN};
                border-bottom:2px solid {CYAN};}}
            QTableWidget{{background:#0D111C;alternate-background-color:#131929;
                gridline-color:#222B42;color:{TXT};selection-background-color:#263859;}}
            QHeaderView::section{{background:#161F33;color:{CYAN};border:1px solid {GRID};
                padding:5px;font-weight:bold;}}
            QTextEdit{{background:#090C14;color:#C9D6E3;border:1px solid {GRID};
                border-radius:4px;}}
            QScrollArea{{border:none;}}
            """)

        def _build(self):
            central = QWidget(); self.setCentralWidget(central)
            root = QVBoxLayout(central)
            root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)
            banner = QFrame()
            banner.setStyleSheet(f"background:#11182B;border:1px solid {CYAN};"
                                 f"border-radius:6px;padding:6px;")
            bl = QHBoxLayout(banner)
            lbl = QLabel("IT3-WAVEPIXEL v3.1  |  M_tot = S4 x T3 over K=Q(sqrt2,sqrt3,sqrt5)"
                         "  |  ROM 0xE100 = 57600 = 240^2  |  Kummer wall sqrt(13) not in K"
                         "  |  ledger: self-counting")
            lbl.setStyleSheet(f"color:{CYAN};font-weight:bold;border:none;")
            bl.addWidget(lbl)
            for t, f in (("Export tab PNG", self.export_png),
                         ("Export audit CSV", self.export_csv),
                         ("Export ledger JSON", self.export_ledger)):
                b = QPushButton(t); b.clicked.connect(f); bl.addWidget(b)
            root.addWidget(banner)
            split = QSplitter(Qt.Orientation.Horizontal); root.addWidget(split, 1)
            left = QWidget(); lv = QVBoxLayout(left)
            lv.setContentsMargins(4, 4, 4, 4); lv.setSpacing(6)

            g1 = QGroupBox("1. CASCADE INPUT"); q1 = QGridLayout(g1)
            self.preset = QComboBox()
            self.preset.addItems([
                "Even rail: 298Ubn (120 -> 82)", "Odd rail: 297Uue (119 -> 81)",
                "294Og pure alpha (118 -> 82)", "293Ts pure alpha (117 -> 83)",
                "282Cn (112 -> 82)", "250Cm f-cavity (96 -> 82)",
                "Ceiling: 426E172 (172 -> 82)", "238U (92 -> 82; 206Pb needs beta-)"])
            self.preset.currentIndexChanged.connect(self.on_preset)
            q1.addWidget(QLabel("Preset:"), 0, 0); q1.addWidget(self.preset, 0, 1, 1, 3)
            self.sz = QSpinBox(); self.sz.setRange(2, 172); self.sz.setValue(120)
            self.sz.valueChanged.connect(self.on_z)
            self.sa = QSpinBox(); self.sa.setRange(2, 460); self.sa.setValue(298)
            self.ss = QSpinBox(); self.ss.setRange(1, 171); self.ss.setValue(82)
            q1.addWidget(QLabel("Start Z:"), 1, 0); q1.addWidget(self.sz, 1, 1)
            q1.addWidget(QLabel("Mass A:"), 1, 2); q1.addWidget(self.sa, 1, 3)
            q1.addWidget(QLabel("Stop Z:"), 2, 0); q1.addWidget(self.ss, 2, 1)
            run = QPushButton("COMPUTE CASCADE"); run.clicked.connect(self.recompute)
            q1.addWidget(run, 2, 2, 1, 2)
            lv.addWidget(g1)

            g2 = QGroupBox("2. TRANSMUTATION LAB"); q2 = QGridLayout(g2)
            self.src = QLineEdit("208Pb"); self.src.setPlaceholderText("208Pb | Pb-208 | 82,208")
            self.dst = QLineEdit("197Au"); self.dst.setPlaceholderText("197Au | Au-197 | 79,197")
            q2.addWidget(QLabel("Source:"), 0, 0); q2.addWidget(self.src, 0, 1)
            q2.addWidget(QLabel("Target:"), 0, 2); q2.addWidget(self.dst, 0, 3)
            bt = QPushButton("TRANSMUTE"); bt.setObjectName("gold")
            bt.clicked.connect(self.transmute)
            bl2 = QPushButton("Load path into HUD"); bl2.clicked.connect(self.load_path)
            q2.addWidget(bt, 1, 0, 1, 2); q2.addWidget(bl2, 1, 2, 1, 2)
            hq = QHBoxLayout()
            for t, s, d in (("208Pb->197Au", "208Pb", "197Au"),
                            ("226Ra->222Rn", "226Ra", "222Rn"),
                            ("223Fr->219At", "223Fr", "219At"),
                            ("238U->206Pb", "238U", "206Pb")):
                b = QPushButton(t)
                b.clicked.connect(lambda _, s=s, d=d: self._pair(s, d))
                hq.addWidget(b)
            q2.addLayout(hq, 2, 0, 1, 4)
            lv.addWidget(g2)

            g3 = QGroupBox("3. SCRUBBER & RAIL SWITCH"); v3 = QVBoxLayout(g3)
            hb = QHBoxLayout()
            for t, f in (("Prev", self.prev), ("Animate", self.play), ("Next", self.next)):
                b = QPushButton(t); b.clicked.connect(f); hb.addWidget(b)
                if t == "Animate":
                    self.b_play = b
            v3.addLayout(hb)
            self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setMinimum(0)
            self.slider.valueChanged.connect(self.on_slider)
            v3.addWidget(self.slider)
            hb2 = QHBoxLayout()
            b_ec = QPushButton("Toggle EC at Z"); b_ec.setObjectName("pink")
            b_ec.clicked.connect(self.toggle_ec)
            b_cl = QPushButton("Reset pure alpha"); b_cl.clicked.connect(self.clear_ec)
            hb2.addWidget(b_ec); hb2.addWidget(b_cl)
            v3.addLayout(hb2)
            lv.addWidget(g3)

            g4 = QGroupBox("4. REGISTER INSPECTOR (Z,N,M)"); v4 = QVBoxLayout(g4)
            self.tele = QTextEdit(); self.tele.setReadOnly(True); self.tele.setMinimumHeight(230)
            v4.addWidget(self.tele); lv.addWidget(g4, 1)

            g5 = QGroupBox("5. ELEMENTARY CHANNELS"); v5 = QVBoxLayout(g5)
            gc = QGridLayout()
            for i, (t, c) in enumerate([("alpha (-2p,-2n)", "alpha"),
                                        ("beta- (n->p)", "beta_minus"),
                                        ("EC (p+e->n)", "ec"),
                                        ("Pb->Au (-3p,-8n)", "pb_au"),
                                        ("UPC 1n knockout", "upc_n"),
                                        ("Auger (-2e)", "auger")]):
                b = QPushButton(t)
                b.clicked.connect(lambda _, c=c: self.channel(c))
                gc.addWidget(b, i // 2, i % 2)
            v5.addLayout(gc)
            self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setMaximumHeight(130)
            v5.addWidget(self.out); lv.addWidget(g5)
            split.addWidget(left)

            self.tabs = QTabWidget(); split.addWidget(self.tabs); split.setSizes([520, 1200])
            self.f_hud = Figure(figsize=(12, 8.6), dpi=100, facecolor=BG)
            self.c_hud = FigureCanvas(self.f_hud)
            self.tabs.addTab(self.c_hud, "1. HUD telemetry")
            self.f_mat = Figure(figsize=(15, 9.4), dpi=100, facecolor=BG)
            self.c_mat = FigureCanvas(self.f_mat)
            self.c_mat.setMinimumSize(1300, 830)
            sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(self.c_mat)
            self.tabs.addTab(sa, "2. 172-layer matrix")
            tw = QWidget(); tv = QVBoxLayout(tw)
            self.table = QTableWidget(); self.table.setColumnCount(12)
            self.table.setHorizontalHeaderLabels(
                ["Step", "Mode", "Transition", "Z old->new", "XOR", "TFI", "ATFI",
                 "Daughter |S,L,mu>", "W_N", "strain", "dQ_top", "Event / hint"])
            self.table.setAlternatingRowColors(True)
            self.table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.cellClicked.connect(self.on_row)
            tv.addWidget(self.table); self.tabs.addTab(tw, "3. Binary audit table")
            self.f_cern = Figure(figsize=(15, 10), dpi=100, facecolor=BG)
            self.c_cern = FigureCanvas(self.f_cern)
            self.c_cern.setMinimumSize(1300, 880)
            sa2 = QScrollArea(); sa2.setWidgetResizable(True); sa2.setWidget(self.c_cern)
            self.tabs.addTab(sa2, "4. Transmutation & mask lab")

        def _pair(self, s, d):
            self.src.setText(s); self.dst.setText(d); self.transmute()
            self.tabs.setCurrentIndex(3)

        def on_preset(self, i):
            p = [(120, 298, 82), (119, 297, 81), (118, 294, 82), (117, 293, 83),
                 (112, 282, 82), (96, 250, 82), (172, 426, 82), (92, 238, 82)]
            if 0 <= i < len(p):
                z, a, s = p[i]
                self.sz.blockSignals(True); self.sz.setValue(z); self.sz.blockSignals(False)
                self.sa.setMinimum(z); self.sa.setValue(a); self.ss.setValue(s)
                self.ec_switches.clear(); self.recompute()

        def on_z(self, z):
            self.sa.setMinimum(z)
            self.sa.setValue(max(z, WPM.reference_mass(z)))
            if z > 83:
                self.ss.setValue(82 if z % 2 == 0 else 81)

        def recompute(self):
            z, a, s = self.sz.value(), self.sa.value(), self.ss.value()
            if a < z:
                a = z; self.sa.setValue(a)
            if s >= z:
                s = max(1, z - 4); self.ss.setValue(s)
            try:
                ch = AlphaCascadeEngine.trace_chain(z, a, s, self.ec_switches)
            except (RegisterOverflow, PhysicalViolation) as e:
                QMessageBox.warning(self, "Register halt", str(e)); return
            self.records, self.halt_msg = ch.records, ch.halt
            self.idx = 0
            self.slider.blockSignals(True)
            self.slider.setMaximum(max(0, len(self.records) - 1)); self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.fill_table(); self.refresh()

        def refresh(self):
            if not self.records:
                self.tele.setHtml(f"<b>HALT:</b> {self.halt_msg or 'no records'}")
                render_hud(self.f_hud, [], 0); self.c_hud.draw()
                render_matrix(self.f_mat, None, 0); self.c_mat.draw()
                return
            self.idx = max(0, min(self.idx, len(self.records) - 1))
            self.telemetry(self.records[self.idx])
            render_hud(self.f_hud, self.records, self.idx); self.c_hud.draw()
            render_matrix(self.f_mat, self.records, self.idx); self.c_mat.draw()
            self.table.selectRow(self.idx)

        def on_slider(self, v):
            self.idx = v; self.refresh()

        def on_row(self, r, c):
            self.idx = r
            self.slider.blockSignals(True); self.slider.setValue(r)
            self.slider.blockSignals(False)
            self.refresh()

        def prev(self):
            if self.idx > 0:
                self.slider.setValue(self.idx - 1)

        def next(self):
            if self.idx < len(self.records) - 1:
                self.slider.setValue(self.idx + 1)

        def play(self):
            if self.timer.isActive():
                self.timer.stop(); self.b_play.setText("Animate")
            else:
                if self.idx >= len(self.records) - 1:
                    self.idx = 0
                self.timer.start(); self.b_play.setText("Pause")

        def _advance(self):
            if self.idx < len(self.records) - 1:
                self.next()
            else:
                self.timer.stop(); self.b_play.setText("Animate")

        def toggle_ec(self):
            if not self.records:
                return
            z = self.records[self.idx].z_parent
            self.ec_switches.symmetric_difference_update({z})
            self.recompute()

        def clear_ec(self):
            self.ec_switches.clear(); self.recompute()

        def telemetry(self, r):
            fp = WPM.facing_partner(r.z_parent)
            rail = "EVEN rail (LSB=0)" if r.z_daughter % 2 == 0 else "ODD rail (LSB=1)"
            halt = (f"<br><span style='color:{PINK};'><b>CHAIN HALT:</b> {self.halt_msg}</span>"
                    if self.halt_msg else "")
            self.tele.setHtml(f"""
            <div style='font-family:Menlo,Consolas,monospace;font-size:11.5px;line-height:1.35;'>
            <span style='color:{CYAN};font-weight:bold;'>[STEP {r.step_idx}/{len(self.records)}
            :: {r.mode}]</span><br>
            <b>Parent:</b> {r.a_parent}{WPM.get_symbol(r.z_parent)} (Z={r.z_parent},
            N={r.a_parent-r.z_parent}) | {r.coord_parent}<br>
            <b>Daughter:</b> {r.a_daughter}{WPM.get_symbol(r.z_daughter)} (Z={r.z_daughter},
            N={r.a_daughter-r.z_daughter}) | {r.coord_daughter}<br>
            <b>Rail:</b> <span style='color:{AMBER};'>{rail}</span>
            <hr style='border:1px solid {GRID};'>
            <b>Z:</b> <code>{r.bin_parent}</code> -> <code>{r.bin_daughter}</code><br>
            <b>XOR:</b> <code style='color:{CYAN};'>{r.bin_xor}</code> (0x{r.xor_mask:02X})<br>
            <b>TFI = {r.tfi}</b> | <b>ATFI = {r.atfi:.3f}</b><br>
            <b>F(Z) = {WPM.get_symbol(fp)}({fp})</b> | W_N = <code>{r.facing_word_hex}</code>
            <hr style='border:1px solid {GRID};'>
            <b>strain Z^2/A:</b> {r.strain_before:.2f} -> {r.strain_after:.2f}<br>
            <b>asym (N-Z)/A:</b> {r.asym_after:.4f}<br>
            <b>unrelaxed Q_top:</b> {r.q_top_unrelaxed:+d}<br>
            <b>Event:</b> <span style='color:{AMBER};'>{r.event_tag}</span><br>
            <b>Hint:</b> <span style='color:{CYAN};'>{r.tunnel_hint}</span>{halt}
            </div>""")

        def channel(self, code):
            if self.records:
                z0 = self.records[self.idx].z_parent
                a0 = self.records[self.idx].a_parent
            else:
                z0, a0 = self.sz.value(), self.sa.value()
            if code == "pb_au":
                self._pair("208Pb", "197Au"); return
            table = {"alpha": (-2, -2, 0, "alpha: dZ=-2, dN=-2"),
                     "beta_minus": (1, -1, 0, "beta-: hardware-first edit"),
                     "ec": (-1, 1, -1, "EC: synchronized HW/SW edit"),
                     "upc_n": (0, -1, 0, "UPC neutron knockout: pure N edit"),
                     "auger": (0, 0, -2, "Auger: two-bit software deletion")}
            dz, dn, dw, title = table[code]
            n0 = a0 - z0
            z1, n1 = z0 + dz, n0 + dn
            if not (1 <= z1 <= 172):
                self.out.setHtml(f"<b>{title}</b><br><span style='color:{PINK};'>"
                                 f"KUMMER / REGISTER HALT: Z'={z1} outside [1,172].</span>")
                return
            if n1 < 0 or (code == "alpha" and n0 < 2):
                self.out.setHtml(f"<b>{title}</b><br><span style='color:{PINK};'>"
                                 f"PHYSICAL HALT: N={n0} insufficient (alpha needs N>=2).</span>")
                return
            a1 = z1 + n1
            mask, tfi = BinaryFrictionALU.compute_tfi(z0, z1)
            atfi, _ = BinaryFrictionALU.compute_atfi(z0, z1)
            self.out.setHtml(
                f"<b>{title}</b><br>({z0},{n0},M_{z0}) -> ({z1},{n1},M_{z0+dw})<br>"
                f"XOR=0x{mask:02X} | TFI={tfi} | ATFI={atfi:.2f}<br>"
                f"dQ_top = dZ({dz:+d}) - dwt(M)({dw:+d}) = "
                f"<span style='color:{PINK};'><b>{dz-dw:+d}</b></span>")

        def transmute(self):
            try:
                z1, a1 = WPM.parse_nuclide_input(self.src.text())
                z2, a2 = WPM.parse_nuclide_input(self.dst.text())
                res = AlphaCascadeEngine.solve_custom_transmutation(z1, a1, z2, a2)
            except (ValueError, PhysicalViolation) as e:
                QMessageBox.warning(self, "Input / register error", str(e)); return
            self.last = res
            lbl = "2-adic TFI" if (res.z1 - res.z2 == 2) else "Hamming wt(Z1^Z2)"
            splits = "; ".join(f"k={k}: a x{k}+p x{p}+n x{n}"
                               for k, p, n in res.nucleon_splits) or "N/A"
            self.out.setHtml(
                f"<span style='color:{AMBER};font-weight:bold;'>{res.direct_equation}</span><br>"
                f"XOR=0x{res.xor_addr:02X} | <b>{lbl}={res.tfi_addr}</b> | "
                f"ATFI^2={res.atfi_sq} ({res.atfi_total:.3f})<br>"
                f"{res.nibble_report}<br>splits: {splits}<br>{res.reachability_note}<br>"
                f"Q_top={res.q_top_unrelaxed:+d} | mask diff={res.mask_xor_weight} bits | "
                f"path <b>{'VALID' if res.path_valid else 'INVALID'}</b>")
            render_cern(self.f_cern, res); self.c_cern.draw()

        def load_path(self):
            if not self.last or not self.last.stepwise_records:
                self.transmute()
            if self.last and self.last.stepwise_records:
                self.records = self.last.stepwise_records
                self.halt_msg = ""
                self.idx = 0
                self.slider.blockSignals(True)
                self.slider.setMaximum(len(self.records) - 1); self.slider.setValue(0)
                self.slider.blockSignals(False)
                self.fill_table(); self.refresh(); self.tabs.setCurrentIndex(0)

        def fill_table(self):
            self.table.setRowCount(len(self.records))
            for i, r in enumerate(self.records):
                ev = r.event_tag + (f" | {r.tunnel_hint}" if r.tunnel_hint != "-" else "")
                cols = [str(r.step_idx), r.mode,
                        f"{r.a_parent}{WPM.get_symbol(r.z_parent)} -> "
                        f"{r.a_daughter}{WPM.get_symbol(r.z_daughter)}",
                        f"{r.bin_parent} -> {r.bin_daughter}",
                        f"{r.bin_xor} (0x{r.xor_mask:02X})", str(r.tfi), f"{r.atfi:.2f}",
                        r.coord_daughter, r.facing_word_hex, f"{r.strain_after:.2f}",
                        f"{r.q_top_unrelaxed:+d}", ev]
                wall = r.tfi >= 3
                for c, v in enumerate(cols):
                    it = QTableWidgetItem(v)
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter if c < 11
                                        else Qt.AlignmentFlag.AlignLeft |
                                        Qt.AlignmentFlag.AlignVCenter)
                    if wall:
                        it.setForeground(QColor(PINK if c in (4, 5, 11) else AMBER))
                        f = it.font(); f.setBold(True); it.setFont(f)
                    self.table.setItem(i, c, it)

        def export_png(self):
            i = self.tabs.currentIndex()
            if i == 2:
                self.export_csv(); return
            fig = {0: self.f_hud, 1: self.f_mat, 3: self.f_cern}[i]
            z0 = self.records[0].z_parent if self.records else 0
            path, _ = QFileDialog.getSaveFileName(
                self, "Save PNG", f"IT3_v3_tab{i+1}_Z{z0}.png", "PNG (*.png)")
            if path:
                fig.savefig(path, dpi=180, facecolor=BG)
                QMessageBox.information(self, "Export", f"saved:\n{path}")

        def export_csv(self):
            if not self.records:
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", f"IT3_v3_audit_Z{self.records[0].z_parent}.csv", "CSV (*.csv)")
            if not path:
                return
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["step", "mode", "Zp", "Ap", "Zd", "Ad", "bin_p", "bin_d",
                            "xor", "TFI", "ATFI", "coord_d", "W_N", "strain_d",
                            "dQ_top", "event", "hint"])
                for r in self.records:
                    w.writerow([r.step_idx, r.mode, r.z_parent, r.a_parent, r.z_daughter,
                                r.a_daughter, r.bin_parent, r.bin_daughter,
                                f"0x{r.xor_mask:02X}", r.tfi, f"{r.atfi:.4f}",
                                r.coord_daughter, r.facing_word_hex, f"{r.strain_after:.4f}",
                                r.q_top_unrelaxed, r.event_tag, r.tunnel_hint])
            QMessageBox.information(self, "Export", f"saved:\n{path}")

        def export_ledger(self):
            path, _ = QFileDialog.getSaveFileName(
                self, "Save ledger", "IT3_v3_ledger.json", "JSON (*.json)")
            if path:
                build_ledger().to_json(path)
                QMessageBox.information(self, "Export", f"ledger saved:\n{path}")


def main() -> None:
    p = argparse.ArgumentParser(description=f"IT3-WavePixel v{VERSION} hardened suite")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--transmute", nargs=2, metavar=("SRC", "DST"))
    p.add_argument("--batch-export", metavar="DIR")
    p.add_argument("--ledger", metavar="JSON")
    p.add_argument("--firewall", action="store_true")
    p.add_argument("--version", action="version", version=f"IT3-WavePixel {VERSION}")
    args = p.parse_args()
    if args.self_test:
        sys.exit(run_self_test(args.ledger))
    if args.ledger:
        build_ledger().to_json(args.ledger); print(f"ledger -> {args.ledger}"); sys.exit(0)
    if args.firewall:
        print_diagnostics(); sys.exit(0)
    if args.transmute:
        sys.exit(cli_transmute(*args.transmute))
    if args.batch_export:
        sys.exit(batch_export(args.batch_export))
    if not HAS_PYQT:
        print("[!] PyQt6 not available -> CLI ledger instead.\n")
        sys.exit(run_self_test())
    app = QApplication(sys.argv)
    win = IT3Suite(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()