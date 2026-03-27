#!/usr/bin/env python3
"""
tools/build_database.py
-----------------------
Generate a comprehensive Raman spectroscopy compound database from curated
literature data. Outputs JSON in the app's expected format.

Usage:
    python tools/build_database.py                          # writes data/rruff_database.json
    python tools/build_database.py --output custom.json     # writes to custom path
"""

import argparse
import json
import os
import sys


def _p(wn, assignment, ri):
    """Shorthand peak constructor."""
    return {"Wavenumber": wn, "Assignment": assignment, "RelativeIntensity": ri}


def _c(name, peaks):
    """Shorthand compound constructor."""
    return {"Name": name, "Peaks": peaks}


def build_minerals():
    """Mineral compounds — sourced from RRUFF reference data and mineralogy literature."""
    return [
        _c("Calcite", [_p(156,"lattice mode",0.3),_p(282,"lattice mode",0.2),_p(713,"v4 CO3 bend",0.15),_p(1085,"v1 CO3 symmetric",1.0),_p(1435,"v3 CO3 asymmetric",0.05)]),
        _c("Quartz", [_p(128,"lattice mode",0.5),_p(206,"lattice mode",0.8),_p(264,"lattice mode",0.15),_p(356,"lattice mode",0.1),_p(464,"Si-O-Si symmetric bend",1.0),_p(695,"Si-O-Si bend",0.1),_p(808,"Si-O stretch",0.05),_p(1162,"Si-O-Si stretch",0.15)]),
        _c("Aragonite", [_p(155,"lattice mode",0.4),_p(207,"lattice mode",0.3),_p(706,"v4 CO3",0.2),_p(1085,"v1 CO3 symmetric",1.0),_p(1462,"v3 CO3",0.06)]),
        _c("Gypsum", [_p(415,"SO4 bending",0.3),_p(493,"SO4 bending",0.2),_p(619,"SO4 bending",0.2),_p(1008,"SO4 symmetric stretch",1.0),_p(1135,"SO4 asymmetric stretch",0.15),_p(3403,"O-H stretch",0.5),_p(3494,"O-H stretch",0.4)]),
        _c("Dolomite", [_p(176,"lattice mode",0.25),_p(300,"lattice mode",0.2),_p(725,"v4 CO3 bend",0.15),_p(1098,"v1 CO3 symmetric",1.0),_p(1443,"v3 CO3 asymmetric",0.05)]),
        _c("Magnetite", [_p(306,"T2g",0.3),_p(540,"T2g",0.5),_p(670,"A1g",1.0)]),
        _c("Hematite", [_p(225,"A1g",0.6),_p(245,"Eg",0.4),_p(292,"Eg",1.0),_p(410,"Eg",0.5),_p(500,"A1g",0.3),_p(611,"Eg",0.4),_p(660,"disorder",0.2),_p(1320,"2-magnon",0.3)]),
        _c("Rutile", [_p(143,"B1g",0.1),_p(235,"multi-phonon",0.15),_p(447,"Eg",1.0),_p(612,"A1g",0.6),_p(826,"B2g",0.05)]),
        _c("Anatase", [_p(144,"Eg",1.0),_p(197,"Eg",0.1),_p(399,"B1g",0.2),_p(513,"A1g",0.15),_p(519,"B1g",0.15),_p(639,"Eg",0.25)]),
        _c("Feldspar (Orthoclase)", [_p(155,"lattice mode",0.3),_p(285,"lattice mode",0.2),_p(475,"Si-O bend",0.4),_p(515,"Si-O-Al bend",1.0),_p(750,"Si-O-Si stretch",0.15),_p(1100,"Si-O stretch",0.1)]),
        _c("Olivine (Forsterite)", [_p(222,"lattice mode",0.15),_p(305,"lattice mode",0.2),_p(434,"SiO4 bend",0.3),_p(546,"SiO4 asymmetric bend",0.2),_p(608,"SiO4 asymmetric bend",0.15),_p(824,"SiO4 symmetric stretch",0.6),_p(856,"SiO4 symmetric stretch",0.9),_p(882,"SiO4 antisymmetric stretch",1.0),_p(965,"SiO4 antisymmetric stretch",0.5)]),
        _c("Pyrite", [_p(343,"Eg",0.8),_p(379,"A1g",1.0),_p(430,"T2g",0.15)]),
        _c("Fluorite", [_p(322,"T2g",1.0)]),
        _c("Barite", [_p(453,"SO4 v2 bend",0.3),_p(462,"SO4 v2 bend",0.35),_p(617,"SO4 v4 bend",0.2),_p(647,"SO4 v4 bend",0.15),_p(988,"SO4 v1 symmetric stretch",1.0),_p(1085,"SO4 v3 asymmetric stretch",0.1),_p(1105,"SO4 v3 asymmetric stretch",0.12),_p(1141,"SO4 v3 asymmetric stretch",0.1)]),
        _c("Apatite", [_p(430,"v2 PO4",0.2),_p(449,"v2 PO4",0.15),_p(581,"v4 PO4",0.25),_p(592,"v4 PO4",0.3),_p(609,"v4 PO4",0.2),_p(962,"v1 PO4 symmetric stretch",1.0),_p(1030,"v3 PO4",0.15),_p(1048,"v3 PO4",0.2),_p(1077,"v3 PO4",0.12)]),
        _c("Talc", [_p(196,"lattice mode",0.3),_p(363,"Mg-O stretch",0.4),_p(466,"Si-O bend",0.2),_p(678,"Si-O-Si symmetric stretch",1.0),_p(1050,"Si-O stretch",0.15),_p(3677,"O-H stretch",0.6)]),
        _c("Muscovite", [_p(198,"lattice mode",0.2),_p(264,"lattice mode",0.3),_p(412,"Al-O stretch",0.4),_p(701,"Si-O-Si stretch",1.0),_p(908,"AlOH deformation",0.15),_p(3628,"O-H stretch",0.5)]),
        _c("Garnet (Almandine)", [_p(170,"T lattice",0.3),_p(215,"R lattice",0.2),_p(346,"R(SiO4)",0.35),_p(557,"v4 SiO4",0.2),_p(632,"v4 SiO4",0.15),_p(870,"v1 SiO4",0.4),_p(918,"v3 SiO4",1.0)]),
        _c("Zircon", [_p(202,"lattice mode",0.15),_p(225,"Eg",0.2),_p(357,"Eg",0.25),_p(439,"A1g",0.3),_p(974,"B1g",0.6),_p(1008,"A1g SiO4 stretch",1.0)]),
        _c("Topaz", [_p(239,"lattice mode",0.2),_p(287,"lattice mode",0.3),_p(335,"lattice mode",0.25),_p(455,"Al-F stretch",0.4),_p(924,"Al-O-Si",0.5),_p(982,"Si-O stretch",1.0),_p(3649,"O-H stretch",0.3)]),
        _c("Corundum", [_p(378,"Eg",0.35),_p(418,"A1g",1.0),_p(432,"Eg",0.5),_p(451,"Eg",0.3),_p(578,"Eg",0.25),_p(645,"A1g",0.15),_p(751,"Eg",0.15)]),
        _c("Siderite", [_p(186,"lattice mode",0.3),_p(287,"lattice mode",0.2),_p(731,"v4 CO3",0.15),_p(1088,"v1 CO3 symmetric",1.0),_p(1445,"v3 CO3",0.05)]),
        _c("Magnesite", [_p(213,"lattice mode",0.3),_p(329,"lattice mode",0.2),_p(739,"v4 CO3",0.15),_p(1094,"v1 CO3 symmetric",1.0),_p(1444,"v3 CO3",0.05)]),
        _c("Anhydrite", [_p(417,"v2 SO4",0.3),_p(500,"v2 SO4",0.15),_p(611,"v4 SO4",0.25),_p(629,"v4 SO4",0.2),_p(676,"v4 SO4",0.1),_p(1018,"v1 SO4 symmetric stretch",1.0),_p(1110,"v3 SO4",0.15),_p(1131,"v3 SO4",0.12),_p(1161,"v3 SO4",0.1)]),
    ]


def build_organics():
    """Common organic compounds — from NIST and spectroscopy handbooks."""
    return [
        _c("Ethanol", [_p(434,"C-C-O deformation",0.2),_p(880,"C-C stretch",0.5),_p(1052,"C-O stretch",0.8),_p(1090,"C-O stretch",1.0),_p(1275,"CH2 wag",0.15),_p(1454,"CH2 scissor",0.3),_p(2876,"CH2 symmetric stretch",0.6),_p(2928,"CH3 asymmetric stretch",0.8)]),
        _c("Methanol", [_p(1033,"C-O stretch",1.0),_p(1108,"CH3 rock",0.3),_p(1454,"CH3 deformation",0.4),_p(2835,"CH3 symmetric stretch",0.7),_p(2945,"CH3 asymmetric stretch",0.8)]),
        _c("Acetone", [_p(390,"C-C-C deformation",0.15),_p(530,"C-C-C deformation",0.1),_p(788,"C-C stretch",0.5),_p(1067,"CH3 rock",0.2),_p(1222,"C-C-C asymmetric stretch",0.3),_p(1430,"CH3 deformation",0.4),_p(1710,"C=O stretch",1.0),_p(2924,"CH3 stretch",0.6),_p(3005,"CH3 asymmetric stretch",0.5)]),
        _c("Benzene", [_p(606,"ring deformation",0.3),_p(849,"C-H out-of-plane",0.3),_p(992,"ring breathing",1.0),_p(1178,"C-H in-plane bend",0.1),_p(1585,"C=C stretch",0.2),_p(3062,"=C-H stretch",0.15)]),
        _c("Toluene", [_p(521,"ring deformation",0.15),_p(786,"C-H out-of-plane",0.25),_p(1002,"ring breathing",1.0),_p(1030,"C-H in-plane",0.3),_p(1210,"C-C stretch",0.1),_p(1379,"CH3 deformation",0.15),_p(1605,"C=C stretch",0.2),_p(2920,"CH3 stretch",0.3),_p(3057,"=C-H stretch",0.15)]),
        _c("Chloroform", [_p(262,"C-Cl bend",0.3),_p(366,"C-Cl symmetric stretch",1.0),_p(668,"C-Cl asymmetric stretch",0.6),_p(761,"C-Cl stretch",0.4),_p(3019,"C-H stretch",0.2)]),
        _c("Acetic Acid", [_p(624,"O-C=O deformation",0.2),_p(891,"C-C stretch",0.5),_p(1049,"C-O stretch",0.3),_p(1294,"C-O stretch",0.4),_p(1430,"CH3 deformation",0.35),_p(1712,"C=O stretch",1.0),_p(2940,"CH3 stretch",0.5)]),
        _c("Dimethyl Sulfoxide", [_p(308,"C-S-C deformation",0.2),_p(383,"C-S stretch",0.3),_p(668,"C-S symmetric stretch",0.6),_p(699,"C-S asymmetric stretch",0.4),_p(953,"CH3 rock",0.2),_p(1042,"S=O stretch",1.0),_p(1420,"CH3 deformation",0.3),_p(2913,"CH3 symmetric stretch",0.5),_p(3000,"CH3 asymmetric stretch",0.4)]),
        _c("Cyclohexane", [_p(384,"ring deformation",0.15),_p(801,"CH2 rock",0.6),_p(1028,"C-C stretch",0.5),_p(1157,"CH2 wag",0.2),_p(1266,"CH2 twist",0.4),_p(1444,"CH2 scissor",0.5),_p(2853,"CH2 symmetric stretch",1.0),_p(2923,"CH2 asymmetric stretch",0.8)]),
        _c("Naphthalene", [_p(395,"ring deformation",0.15),_p(513,"ring deformation",0.2),_p(764,"C-H out-of-plane",0.6),_p(1021,"ring breathing",0.3),_p(1148,"C-H in-plane",0.15),_p(1382,"ring stretch",1.0),_p(1464,"ring stretch",0.25),_p(1577,"C=C stretch",0.15),_p(3057,"C-H stretch",0.2)]),
        _c("Formic Acid", [_p(625,"O-C=O deformation",0.2),_p(1060,"C-O stretch",0.4),_p(1220,"C-H in-plane bend",0.3),_p(1380,"C-O stretch",0.5),_p(1724,"C=O stretch",1.0),_p(2869,"C-H stretch",0.5)]),
        _c("Isopropanol", [_p(819,"C-C-O stretch",0.5),_p(954,"CH3 rock",0.3),_p(1128,"C-O stretch",0.6),_p(1340,"C-H bend",0.2),_p(1454,"CH3 deformation",0.4),_p(2882,"CH stretch",0.7),_p(2920,"CH3 symmetric stretch",0.8),_p(2972,"CH3 asymmetric stretch",1.0)]),
        _c("Glycerol", [_p(485,"C-C-O deformation",0.15),_p(820,"C-C stretch",0.3),_p(922,"C-O stretch",0.4),_p(1055,"C-O stretch",0.6),_p(1112,"C-O stretch",0.5),_p(1260,"C-H bend",0.2),_p(1462,"CH2 deformation",0.3),_p(2880,"C-H stretch",0.7),_p(2945,"C-H stretch",1.0)]),
        _c("Urea", [_p(547,"N-C-N deformation",0.2),_p(1003,"C-N stretch",0.5),_p(1465,"N-C-N asymmetric stretch",0.3),_p(1541,"NH2 deformation",0.4),_p(1630,"C=O stretch",1.0),_p(3348,"N-H symmetric stretch",0.5),_p(3435,"N-H asymmetric stretch",0.4)]),
        _c("Oxalic Acid", [_p(477,"C-C=O deformation",0.2),_p(596,"O-C=O bend",0.15),_p(861,"C-C stretch",0.5),_p(1189,"C-O stretch",0.3),_p(1456,"C-O stretch",0.4),_p(1741,"C=O stretch",1.0)]),
    ]


def build_pharmaceuticals():
    """Pharmaceutical compounds — from published Raman reference spectra."""
    return [
        _c("Paracetamol", [_p(329,"ring torsion",0.15),_p(503,"ring deformation",0.2),_p(651,"ring deformation",0.15),_p(711,"N-H wag",0.3),_p(797,"C-H out-of-plane",0.25),_p(834,"C-H out-of-plane",0.35),_p(858,"ring breathing",1.0),_p(1105,"C-N stretch",0.35),_p(1169,"C-H in-plane bend",0.25),_p(1236,"C-O stretch",0.5),_p(1324,"C-N stretch amide",0.6),_p(1562,"amide II",0.4),_p(1610,"ring C=C stretch",0.55),_p(1648,"C=O amide stretch",0.9),_p(3326,"N-H stretch",0.4)]),
        _c("Aspirin", [_p(565,"ring deformation",0.2),_p(750,"C-H out-of-plane",0.4),_p(784,"ring breathing",0.3),_p(1045,"C-O stretch",0.3),_p(1190,"C-O stretch",0.5),_p(1309,"C-C stretch",0.5),_p(1607,"C=C stretch",0.6),_p(1752,"C=O stretch (ester)",1.0),_p(3072,"=C-H stretch",0.2)]),
        _c("Ibuprofen", [_p(636,"ring deformation",0.2),_p(747,"C-H out-of-plane",0.25),_p(815,"aromatic C-H",0.35),_p(960,"C-C stretch",0.2),_p(1006,"ring breathing",0.4),_p(1184,"C-O stretch",0.3),_p(1230,"C-H bend",0.35),_p(1382,"CH3 symmetric deformation",0.4),_p(1456,"CH3 asymmetric deformation",0.45),_p(1608,"C=C stretch",0.5),_p(1660,"C=O stretch",1.0),_p(2870,"CH3 symmetric stretch",0.5),_p(2933,"CH3 asymmetric stretch",0.55)]),
        _c("Caffeine", [_p(556,"ring deformation",0.2),_p(644,"ring breathing",0.25),_p(741,"C-N stretch",0.3),_p(925,"N-CH3 stretch",0.2),_p(1071,"C-N stretch",0.15),_p(1239,"C-N stretch",0.25),_p(1328,"imidazole stretch",0.5),_p(1403,"C-N stretch",0.35),_p(1555,"C=N stretch",0.4),_p(1600,"ring stretch",0.3),_p(1659,"C=O stretch",1.0),_p(1698,"C=O stretch",0.85)]),
        _c("Naproxen", [_p(389,"ring deformation",0.15),_p(511,"ring deformation",0.2),_p(745,"C-H out-of-plane",0.25),_p(860,"C-H out-of-plane",0.2),_p(1029,"ring breathing",0.3),_p(1161,"C-O-C stretch",0.25),_p(1262,"C-O stretch",0.35),_p(1390,"CH3 deformation",0.4),_p(1482,"ring C=C stretch",0.3),_p(1631,"ring C=C stretch",0.55),_p(1685,"C=O stretch",1.0)]),
        _c("Amoxicillin", [_p(629,"ring deformation",0.2),_p(777,"C-S stretch",0.3),_p(1004,"ring breathing",0.25),_p(1138,"C-N stretch",0.3),_p(1252,"amide III",0.35),_p(1390,"COO- symmetric stretch",0.4),_p(1512,"amide II",0.45),_p(1615,"C=C stretch",0.5),_p(1668,"C=O beta-lactam",1.0),_p(1775,"C=O ester",0.3)]),
        _c("Diclofenac", [_p(439,"ring deformation",0.15),_p(571,"C-Cl stretch",0.25),_p(680,"C-Cl stretch",0.3),_p(769,"C-H out-of-plane",0.25),_p(953,"C-H out-of-plane",0.2),_p(1045,"C-N stretch",0.3),_p(1167,"C-H in-plane",0.25),_p(1285,"C-N stretch",0.4),_p(1453,"CH2 scissor",0.3),_p(1507,"NH bend",0.35),_p(1579,"C=C stretch",0.8),_p(1605,"C=C stretch",0.6),_p(1685,"C=O stretch",1.0)]),
        _c("Metformin", [_p(700,"C-N stretch",0.3),_p(740,"N-H wag",0.25),_p(936,"C-N stretch",0.35),_p(1065,"C-N stretch",0.4),_p(1170,"N-H bend",0.3),_p(1275,"C-N stretch",0.45),_p(1418,"CH3 deformation",0.35),_p(1473,"C=N stretch",0.5),_p(1558,"C=N stretch",0.55),_p(1625,"NH2 scissor",0.4),_p(1668,"C=N stretch",1.0),_p(3170,"N-H stretch",0.3),_p(3370,"N-H stretch",0.35)]),
        _c("Ciprofloxacin", [_p(518,"ring deformation",0.15),_p(705,"C-F stretch",0.3),_p(785,"ring breathing",0.25),_p(1028,"C-N stretch",0.2),_p(1254,"C-F stretch",0.3),_p(1340,"quinolone stretch",0.5),_p(1386,"COO stretch",0.6),_p(1474,"ring stretch",0.4),_p(1548,"ring C=C stretch",0.45),_p(1622,"C=O stretch",0.7),_p(1708,"COOH C=O stretch",1.0)]),
        _c("Acetaminophen (Form II)", [_p(390,"ring torsion",0.1),_p(465,"ring deformation",0.15),_p(505,"ring deformation",0.2),_p(652,"ring out-of-plane",0.15),_p(797,"C-H oop",0.25),_p(835,"C-H oop",0.35),_p(855,"ring breathing",0.95),_p(1170,"C-H in-plane",0.25),_p(1237,"C-O stretch",0.5),_p(1325,"amide III",0.6),_p(1612,"C=C stretch",0.55),_p(1655,"C=O stretch",1.0)]),
        _c("Celecoxib", [_p(585,"ring deformation",0.2),_p(767,"C-H out-of-plane",0.25),_p(1004,"ring breathing",0.35),_p(1094,"C-F stretch",0.3),_p(1166,"C-H in-plane",0.25),_p(1228,"C-N stretch",0.4),_p(1348,"SO2 symmetric stretch",0.6),_p(1446,"ring stretch",0.35),_p(1524,"ring C=C stretch",0.45),_p(1597,"ring C=C stretch",0.55),_p(1728,"C=N stretch",1.0)]),
        _c("Omeprazole", [_p(564,"ring deformation",0.2),_p(786,"C-H out-of-plane",0.25),_p(834,"C-H out-of-plane",0.2),_p(1011,"ring breathing",0.3),_p(1147,"S=O stretch",0.5),_p(1290,"C-O stretch",0.35),_p(1370,"CH3 deformation",0.3),_p(1468,"ring stretch",0.4),_p(1579,"ring C=C/C=N",0.6),_p(1609,"ring C=C stretch",0.7),_p(1689,"C=N stretch",1.0)]),
    ]


def build_polymers():
    """Common polymers for Raman identification."""
    return [
        _c("Polystyrene", [_p(621,"ring deformation",0.3),_p(795,"C-H out-of-plane",0.2),_p(1001,"ring breathing",1.0),_p(1031,"C-H in-plane",0.4),_p(1155,"C-C stretch",0.15),_p(1450,"CH2 scissor",0.2),_p(1583,"C=C stretch",0.25),_p(1602,"C=C stretch",0.35),_p(2852,"CH2 symmetric stretch",0.3),_p(2904,"CH2 asymmetric stretch",0.3),_p(3054,"=C-H stretch",0.3),_p(3082,"=C-H stretch",0.2)]),
        _c("Polyethylene", [_p(1063,"C-C stretch (trans)",0.3),_p(1130,"C-C stretch (gauche)",0.4),_p(1170,"CH2 wag",0.15),_p(1296,"CH2 twist",0.6),_p(1440,"CH2 scissor",0.5),_p(2850,"CH2 symmetric stretch",1.0),_p(2883,"CH2 asymmetric stretch",0.9)]),
        _c("Polypropylene", [_p(399,"CH2 wag",0.15),_p(808,"C-C stretch / CH2 rock",0.5),_p(841,"C-C stretch / CH2 rock",0.6),_p(973,"CH3 rock",0.3),_p(998,"C-C stretch",0.2),_p(1152,"C-C stretch / C-H bend",0.3),_p(1330,"CH2 deformation",0.2),_p(1460,"CH3 deformation",0.5),_p(2840,"CH stretch",0.4),_p(2870,"CH2 symmetric stretch",0.6),_p(2920,"CH2 asymmetric stretch",0.8),_p(2953,"CH3 asymmetric stretch",1.0)]),
        _c("PET (Polyethylene Terephthalate)", [_p(633,"ring deformation",0.2),_p(795,"ring C-H oop",0.15),_p(858,"ring breathing",0.25),_p(1096,"C-O stretch",0.3),_p(1117,"C-O stretch",0.35),_p(1287,"ring C-C stretch",0.4),_p(1614,"ring C=C stretch",0.45),_p(1726,"C=O stretch",1.0),_p(3080,"ring C-H stretch",0.15)]),
        _c("Nylon-6", [_p(934,"C-C stretch",0.3),_p(1065,"C-C stretch (trans)",0.25),_p(1130,"C-C stretch",0.3),_p(1303,"amide III",0.35),_p(1442,"CH2 scissor",0.45),_p(1636,"amide I (C=O stretch)",1.0),_p(2857,"CH2 symmetric stretch",0.5),_p(2900,"CH2 asymmetric stretch",0.55),_p(3303,"N-H stretch",0.4)]),
        _c("PMMA (Polymethyl Methacrylate)", [_p(600,"C-C-O deformation",0.15),_p(813,"C-O-C stretch",0.2),_p(966,"OCH3 rock",0.15),_p(1124,"C-O stretch",0.2),_p(1240,"C-O-C stretch",0.3),_p(1450,"CH3 deformation",0.4),_p(1731,"C=O stretch",1.0),_p(2843,"O-CH3 stretch",0.3),_p(2951,"C-H stretch",0.6)]),
        _c("PVC (Polyvinyl Chloride)", [_p(358,"C-Cl stretch",0.3),_p(636,"C-Cl stretch",0.5),_p(694,"C-Cl stretch",0.45),_p(1100,"C-C stretch",0.15),_p(1176,"C-H wag",0.2),_p(1333,"CH2 deformation",0.25),_p(1434,"CH2 scissor",0.4),_p(2853,"CH2 symmetric stretch",0.8),_p(2912,"CH2 asymmetric stretch",1.0)]),
        _c("PTFE (Teflon)", [_p(292,"CF2 deformation",0.2),_p(385,"CF2 wag",0.3),_p(576,"CF2 deformation",0.15),_p(731,"CF2 symmetric stretch",1.0),_p(1218,"CF2 asymmetric stretch",0.4),_p(1301,"CF2 stretch",0.35),_p(1382,"CF2 wag",0.2)]),
        _c("Polyurethane", [_p(637,"ring deformation",0.15),_p(775,"C-H out-of-plane",0.2),_p(1065,"C-O-C stretch",0.3),_p(1186,"C-O-C stretch",0.25),_p(1307,"amide III",0.3),_p(1440,"CH2 scissor",0.35),_p(1533,"amide II",0.4),_p(1616,"ring C=C stretch",0.45),_p(1700,"C=O stretch",1.0),_p(2862,"CH2 symmetric stretch",0.45),_p(2940,"CH2 asymmetric stretch",0.5),_p(3330,"N-H stretch",0.3)]),
        _c("Polycarbonate", [_p(637,"ring deformation",0.2),_p(707,"ring deformation",0.15),_p(830,"ring C-H oop",0.25),_p(887,"ring breathing",0.3),_p(1006,"ring breathing",0.35),_p(1112,"C-O stretch",0.25),_p(1234,"C-O-C asymmetric stretch",0.4),_p(1456,"ring C=C stretch",0.3),_p(1602,"ring C=C stretch",0.5),_p(1775,"C=O stretch",1.0),_p(3060,"ring C-H stretch",0.15)]),
        _c("ABS (Acrylonitrile Butadiene Styrene)", [_p(620,"ring deformation",0.2),_p(760,"C-H out-of-plane",0.15),_p(1002,"styrene ring breathing",1.0),_p(1032,"C-H in-plane",0.3),_p(1157,"butadiene C-C stretch",0.15),_p(1452,"CH2 scissor",0.25),_p(1583,"styrene C=C",0.2),_p(1602,"styrene C=C",0.3),_p(2240,"C-N stretch (nitrile)",0.4),_p(2852,"CH2 stretch",0.3),_p(3055,"=C-H stretch",0.2)]),
    ]


def build_carbon_materials():
    """Carbon allotropes, graphene, CNTs, etc."""
    return [
        _c("Graphite", [_p(1350,"D band",0.2),_p(1582,"G band",1.0),_p(2450,"D+D'' band",0.05),_p(2700,"2D band",0.6),_p(3248,"2D' band",0.05)]),
        _c("Diamond", [_p(1332,"C-C symmetric stretch",1.0)]),
        _c("Single-Layer Graphene", [_p(1583,"G band",0.4),_p(2680,"2D band",1.0)]),
        _c("Multi-Layer Graphene", [_p(1350,"D band",0.1),_p(1583,"G band",0.7),_p(2700,"2D band",1.0)]),
        _c("Carbon Nanotube (SWCNT)", [_p(165,"RBM",0.5),_p(1340,"D band",0.15),_p(1590,"G+ band",1.0),_p(1570,"G- band",0.6),_p(2680,"2D band",0.4)]),
        _c("Carbon Nanotube (MWCNT)", [_p(1350,"D band",0.5),_p(1580,"G band",1.0),_p(2700,"2D band",0.3)]),
        _c("Amorphous Carbon", [_p(1350,"D band",0.9),_p(1580,"G band",1.0),_p(2700,"2D band (broad)",0.15)]),
        _c("Graphene Oxide", [_p(1350,"D band",0.95),_p(1600,"G band",1.0),_p(2700,"2D band (broad)",0.1)]),
        _c("Fullerene C60", [_p(272,"Hg(1)",0.3),_p(496,"Ag(1)",0.6),_p(710,"Hg(3)",0.15),_p(774,"Hg(4)",0.25),_p(1099,"Hg(5)",0.15),_p(1248,"Hg(6)",0.2),_p(1425,"Hg(7)",0.3),_p(1469,"Ag(2) pentagonal pinch",1.0),_p(1574,"Hg(8)",0.25)]),
        _c("Diamond-Like Carbon (DLC)", [_p(1332,"sp3 diamond",0.4),_p(1360,"D band (sp2)",0.6),_p(1580,"G band (sp2)",1.0)]),
    ]


def build_oxides_semiconductors():
    """Metal oxides and semiconductor materials."""
    return [
        _c("Silicon", [_p(302,"TA overtone",0.1),_p(520,"first-order LO-TO",1.0),_p(960,"second-order 2TO",0.05)]),
        _c("Zinc Oxide (ZnO)", [_p(99,"E2-low",0.25),_p(332,"E2H-E2L",0.15),_p(380,"A1(TO)",0.2),_p(410,"E1(TO)",0.15),_p(438,"E2-high",1.0),_p(584,"A1(LO)",0.1)]),
        _c("Cerium Oxide (CeO2)", [_p(465,"F2g",1.0),_p(600,"defect-induced",0.15),_p(1170,"2LO",0.05)]),
        _c("Iron Oxide (Maghemite)", [_p(350,"T2g",0.3),_p(500,"Eg",0.4),_p(670,"A1g",1.0),_p(720,"A1g",0.3)]),
        _c("Tin Oxide (SnO2)", [_p(476,"Eg",0.35),_p(634,"A1g",1.0),_p(776,"B2g",0.2)]),
        _c("Tungsten Oxide (WO3)", [_p(133,"lattice mode",0.25),_p(272,"O-W-O bending",0.5),_p(326,"O-W-O bending",0.3),_p(715,"W-O-W stretch",0.8),_p(808,"W=O stretch",1.0)]),
        _c("Molybdenum Disulfide (MoS2)", [_p(286,"E1g",0.2),_p(383,"E2g1",0.7),_p(408,"A1g",1.0),_p(454,"A2u",0.05)]),
        _c("Copper Oxide (CuO)", [_p(296,"Ag",1.0),_p(346,"Bg",0.5),_p(631,"Bg",0.3)]),
        _c("Aluminum Oxide (Al2O3)", [_p(378,"Eg",0.35),_p(418,"A1g",1.0),_p(432,"Eg",0.5),_p(451,"Eg",0.3),_p(578,"Eg",0.25),_p(645,"A1g",0.15),_p(751,"Eg",0.15)]),
        _c("Gallium Nitride (GaN)", [_p(144,"E2-low",0.15),_p(533,"A1(TO)",0.3),_p(560,"E1(TO)",0.25),_p(569,"E2-high",1.0),_p(735,"A1(LO)",0.2),_p(743,"E1(LO)",0.15)]),
        _c("Barium Titanate (BaTiO3)", [_p(185,"A1(LO)",0.25),_p(257,"A1(TO)",0.5),_p(305,"B1/E(TO+LO)",0.6),_p(516,"A1(TO)",1.0),_p(715,"A1(LO)/E(LO)",0.3)]),
        _c("Lithium Niobate (LiNbO3)", [_p(153,"E(TO)",0.2),_p(238,"E(TO)",0.35),_p(264,"A1(TO)",0.45),_p(276,"E(TO)",0.3),_p(324,"E(TO)",0.4),_p(370,"A1(TO)",0.5),_p(432,"E(TO)",0.35),_p(580,"E(TO)",0.3),_p(630,"A1(TO)",1.0),_p(875,"A1(LO)",0.15)]),
        _c("Vanadium Pentoxide (V2O5)", [_p(145,"lattice mode",0.4),_p(195,"lattice mode",0.3),_p(283,"V=O bend",0.45),_p(304,"V-O-V bend",0.35),_p(405,"V-O-V bend",0.3),_p(481,"V-O-V stretch",0.25),_p(527,"V-O-V stretch",0.2),_p(700,"V-O-V stretch",0.5),_p(994,"V=O stretch",1.0)]),
    ]


def build_biomolecules():
    """Biological molecules and amino acids."""
    return [
        _c("L-Alanine", [_p(532,"CO2 rock",0.15),_p(770,"CO2 wag",0.2),_p(846,"C-C-N stretch",0.25),_p(919,"C-C stretch",0.4),_p(1004,"C-C stretch",0.2),_p(1114,"NH3 rock",0.3),_p(1307,"CH bend",0.35),_p(1375,"CH3 symmetric deformation",0.3),_p(1461,"CH3 asymmetric deformation",0.4),_p(1595,"COO- asymmetric stretch",0.5),_p(2934,"C-H stretch",0.6),_p(2989,"CH3 asymmetric stretch",1.0)]),
        _c("L-Cysteine", [_p(500,"S-S stretch",0.15),_p(679,"C-S stretch",0.5),_p(767,"CO2 wag",0.2),_p(862,"C-C-N stretch",0.25),_p(938,"C-C stretch",0.2),_p(1064,"C-N stretch",0.3),_p(1197,"NH3 rock",0.25),_p(1344,"CH bend",0.35),_p(1407,"COO- symmetric stretch",0.4),_p(1583,"COO- asymmetric stretch / NH3 bend",0.5),_p(2551,"S-H stretch",1.0),_p(2960,"C-H stretch",0.4)]),
        _c("L-Phenylalanine", [_p(621,"ring deformation",0.2),_p(750,"ring breathing",0.15),_p(831,"ring C-H oop",0.1),_p(1003,"ring breathing",1.0),_p(1032,"C-H in-plane bend",0.3),_p(1208,"C-C stretch",0.15),_p(1584,"ring C=C stretch",0.2),_p(1606,"ring C=C stretch",0.25),_p(3062,"ring C-H stretch",0.15)]),
        _c("L-Tyrosine", [_p(643,"ring deformation",0.15),_p(829,"ring breathing",1.0),_p(856,"ring breathing (Fermi doublet)",0.6),_p(1177,"C-H in-plane",0.25),_p(1210,"C-O stretch (phenol)",0.3),_p(1268,"ring C-O-H bend",0.2),_p(1447,"ring C=C stretch",0.15),_p(1617,"ring C=C stretch",0.3),_p(3060,"ring C-H stretch",0.1)]),
        _c("L-Tryptophan", [_p(577,"indole ring deformation",0.15),_p(760,"indole ring breathing",0.6),_p(881,"indole N-H wag",0.15),_p(1013,"indole ring",0.35),_p(1236,"C-H bend",0.15),_p(1342,"indole ring stretch",0.4),_p(1462,"indole C=C stretch",0.2),_p(1557,"indole C=C stretch",0.5),_p(1622,"indole ring stretch",1.0)]),
        _c("Glucose", [_p(420,"C-C-C deformation",0.15),_p(525,"C-C-O deformation",0.2),_p(842,"C-H deformation",0.25),_p(911,"C-O stretch",0.3),_p(1022,"C-O stretch",0.4),_p(1065,"C-O stretch",0.45),_p(1127,"C-O-H bend",1.0),_p(1340,"C-O-H bend",0.3),_p(1462,"C-H deformation",0.25),_p(2893,"C-H stretch",0.5),_p(2947,"C-H stretch",0.55)]),
        _c("Sucrose", [_p(404,"C-C-C deformation",0.1),_p(541,"C-C-O deformation",0.15),_p(634,"C-C-O deformation",0.15),_p(840,"C-H deformation",0.2),_p(878,"C-O-C stretch",0.35),_p(921,"C-O stretch",0.25),_p(1063,"C-O stretch",0.4),_p(1133,"C-O-H bend",1.0),_p(1263,"C-O-H bend",0.2),_p(1370,"C-H deformation",0.25),_p(1460,"C-H deformation",0.3),_p(2910,"C-H stretch",0.5),_p(2945,"C-H stretch",0.55)]),
        _c("Amide I (Protein backbone)", [_p(1004,"phenylalanine ring",0.3),_p(1250,"amide III (random coil)",0.3),_p(1268,"amide III (alpha-helix)",0.35),_p(1400,"COO- symmetric stretch",0.2),_p(1450,"CH2 scissor",0.25),_p(1555,"amide II",0.3),_p(1655,"amide I (alpha-helix)",1.0),_p(1670,"amide I (beta-sheet)",0.8),_p(2930,"C-H stretch",0.4)]),
        _c("DNA backbone", [_p(670,"thymine ring",0.2),_p(729,"adenine ring",0.25),_p(784,"cytosine ring / PO2 stretch",0.3),_p(830,"B-form marker",0.35),_p(1013,"sugar-phosphate",0.2),_p(1096,"PO2- symmetric stretch",1.0),_p(1240,"PO2- asymmetric stretch",0.3),_p(1340,"adenine / guanine",0.25),_p(1490,"purine ring",0.15),_p(1578,"purine ring",0.2)]),
        _c("Cholesterol", [_p(428,"ring deformation",0.15),_p(548,"ring deformation",0.1),_p(609,"ring deformation",0.1),_p(700,"C-C steroidal stretch",1.0),_p(882,"C-C stretch",0.2),_p(958,"C-C stretch",0.15),_p(1055,"C-O stretch",0.1),_p(1130,"C-C stretch (trans)",0.15),_p(1440,"CH2 scissor",0.4),_p(1672,"C=C stretch",0.3),_p(2850,"CH2 symmetric stretch",0.5),_p(2870,"CH3 symmetric stretch",0.45),_p(2930,"CH2 asymmetric stretch",0.55),_p(2960,"CH3 asymmetric stretch",0.5)]),
        _c("Hemoglobin", [_p(674,"porphyrin ring (v7)",0.2),_p(752,"porphyrin ring (v15)",0.3),_p(1003,"phenylalanine",0.25),_p(1127,"porphyrin (v22)",0.15),_p(1224,"amide III",0.2),_p(1310,"porphyrin (v21)",0.25),_p(1375,"porphyrin (v4) oxidation marker",1.0),_p(1545,"porphyrin (v11)",0.4),_p(1585,"porphyrin (v37)",0.35),_p(1620,"porphyrin (v(C=C))",0.3),_p(1655,"amide I",0.5)]),
        _c("Lipid (General)", [_p(864,"C-C stretch",0.1),_p(1065,"C-C stretch (trans)",0.25),_p(1130,"C-C stretch (gauche)",0.2),_p(1265,"=C-H in-plane bend",0.3),_p(1301,"CH2 twist",0.35),_p(1440,"CH2 scissor",0.5),_p(1660,"C=C stretch",0.3),_p(2850,"CH2 symmetric stretch",1.0),_p(2885,"CH2 asymmetric stretch",0.9),_p(3010,"=C-H stretch",0.15)]),
    ]


def build_additional():
    """Additional commonly encountered compounds."""
    return [
        _c("Water (liquid)", [_p(1640,"H-O-H bend",0.3),_p(3250,"O-H stretch (symmetric)",0.7),_p(3420,"O-H stretch (asymmetric)",1.0)]),
        _c("Carbon Dioxide (CO2)", [_p(1285,"v1 symmetric stretch / 2v2 Fermi",0.7),_p(1388,"v1 symmetric stretch / 2v2 Fermi",1.0)]),
        _c("Sulfur (elemental)", [_p(152,"lattice mode",0.3),_p(218,"S-S stretch",0.6),_p(246,"S-S stretch",0.4),_p(437,"S-S stretch",0.15),_p(473,"S-S stretch",1.0)]),
        _c("Calcium Carbonate (Vaterite)", [_p(268,"lattice mode",0.3),_p(301,"lattice mode",0.2),_p(739,"v4 CO3",0.15),_p(750,"v4 CO3",0.12),_p(1075,"v1 CO3",0.9),_p(1090,"v1 CO3",1.0)]),
        _c("Sodium Chloride (NaCl)", [_p(250,"2TA",0.2),_p(302,"LO-TO",0.15)]),
        _c("Calcium Sulfate", [_p(417,"v2 SO4",0.25),_p(497,"v2 SO4",0.15),_p(610,"v4 SO4",0.2),_p(628,"v4 SO4",0.15),_p(1017,"v1 SO4 symmetric stretch",1.0),_p(1110,"v3 SO4",0.15),_p(1130,"v3 SO4",0.12)]),
        _c("Titanium Dioxide (Brookite)", [_p(128,"A1g",0.3),_p(153,"A1g",0.5),_p(247,"A1g",0.25),_p(290,"B1g",0.2),_p(322,"B2g",0.35),_p(366,"B3g",0.15),_p(413,"A1g",0.2),_p(502,"A1g",0.15),_p(545,"A1g",0.1),_p(585,"B3g",0.1),_p(636,"A1g",1.0)]),
        _c("Potassium Permanganate", [_p(350,"v2 MnO4",0.2),_p(400,"v4 MnO4",0.25),_p(450,"v4 MnO4",0.15),_p(840,"v1 MnO4 symmetric stretch",1.0),_p(920,"v3 MnO4 asymmetric stretch",0.3)]),
        _c("Sodium Sulfate", [_p(451,"v2 SO4",0.2),_p(619,"v4 SO4",0.25),_p(632,"v4 SO4",0.2),_p(993,"v1 SO4 symmetric stretch",1.0),_p(1101,"v3 SO4",0.15),_p(1131,"v3 SO4",0.12),_p(1153,"v3 SO4",0.1)]),
        _c("Fly Ash (Mullite phase)", [_p(305,"lattice mode",0.2),_p(408,"Al-O stretch",0.3),_p(600,"Si-O-Al stretch",0.4),_p(730,"Si-O-Si stretch",0.3),_p(960,"Si-O stretch",0.5),_p(1080,"Si-O stretch",1.0),_p(1170,"Si-O-Si stretch",0.2)]),
        _c("Calcium Hydroxide (Portlandite)", [_p(255,"Ca-O lattice",0.3),_p(357,"Ca-OH vibration",0.5),_p(680,"O-H libration",0.2),_p(3620,"O-H stretch (sharp)",1.0)]),
        _c("Calcium Silicate Hydrate (C-S-H)", [_p(320,"Ca-O lattice",0.2),_p(450,"Si-O bend",0.3),_p(670,"Si-O-Si bend",1.0),_p(850,"Si-O stretch",0.4),_p(1010,"Si-O stretch",0.5),_p(1085,"CO3 v1 (carbonation)",0.3),_p(3450,"O-H stretch (broad)",0.5)]),
    ]


def build_database():
    """Build the full compound database."""
    return {
        "Minerals": build_minerals(),
        "Organics": build_organics(),
        "Pharmaceuticals": build_pharmaceuticals(),
        "Polymers": build_polymers(),
        "Carbon Materials": build_carbon_materials(),
        "Oxides and Semiconductors": build_oxides_semiconductors(),
        "Biomolecules": build_biomolecules(),
        "Additional Compounds": build_additional(),
    }


def main():
    parser = argparse.ArgumentParser(description="Build Raman compound database")
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "rruff_database.json"),
        help="Output JSON path (default: data/rruff_database.json)",
    )
    args = parser.parse_args()

    db = build_database()

    # Stats
    total_compounds = sum(len(v) for v in db.values())
    total_peaks = sum(len(p["Peaks"]) for compounds in db.values() for p in compounds)
    print(f"Built database: {total_compounds} compounds, {total_peaks} peaks, {len(db)} categories")

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print(f"Written to: {output}")


if __name__ == "__main__":
    main()
