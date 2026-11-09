import numpy as np

METISSE_Z_GRID = np.array(
    [1.4e-05, 2.1e-05, 2.8e-05, 4.2e-05, 5.6e-05, 7.0e-05, 9.8e-05,
     1.4e-04, 2.1e-04, 2.8e-04, 4.2e-04, 5.6e-04, 7.0e-04, 9.8e-04,
     1.4e-03, 2.1e-03, 2.8e-03, 4.2e-03, 5.6e-03, 7.0e-03, 9.8e-03,
     1.4e-02, 2.1e-02, 2.8e-02]#, 4.2e-02, 5.6e-02, 7.0e-02, 9.8e-02]
).round(6)
METISSE_Z_EDGES = 10**(np.concatenate([[-np.inf], 0.5 * (np.log10(METISSE_Z_GRID[1:]) + np.log10(METISSE_Z_GRID[:-1])), [np.inf]]))

SSE_Z_GRID = np.logspace(-4, np.log10(0.03), 50).round(5)
SSE_Z_EDGES = np.concatenate([[0], 0.5 * (SSE_Z_GRID[1:] + SSE_Z_GRID[:-1]), [np.inf]])

VARIATIONS = {
    "sse": {
        "label": "SSE"
    },
    "metisse": {
        "label": "Standard METISSE"
    },
    "acc_lim_0": {
        "label": r"$\beta = 0$"
    },
    "gamma_m3": {
        "label": r"$\gamma = \gamma_{\rm disk}$"
    },
    "acc_lim_0_gamma_m3": {
        "label": r"$\beta = 0, \gamma = \gamma_{\rm disk}$"
    },
    "alpha_10": {
        "label": r"$\alpha_{\rm CE} = 10$"
    },
    "new_winds": {
        "label": r"New winds"
    },
    "new_cbm": {
        "label": r"New CBM"
    },
    "new_winds_and_cbm": {
        "label": r"New winds + New CBM"
    },
    "olejak_20": {
        "label": r"$q_{\rm MT} \ge 20$ stable, non-conservative MT"
    },
    "olejak_40": {
        "label": r"$q_{\rm MT} \ge 40$ stable, non-conservative MT"
    },
    "olejak_20_new_winds": {
        "label": r"New winds + $q_{\rm MT} \ge 20$ stable, non-conservative MT"
    },
    "olejak_20_new_cbm": {
        "label": r"New CBM + $q_{\rm MT} \ge 20$ stable, non-conservative MT"
    },
    "olejak_20_new_winds_and_cbm": {
        "label": r"New winds + New CBM + $q_{\rm MT} \ge 20$ stable, non-conservative MT"
    }
}
for v in VARIATIONS:
    if "v" == "sse":
        VARIATIONS[v]["Z_edges"] = SSE_Z_EDGES
        VARIATIONS[v]["Z_grid"] = SSE_Z_GRID
    else:
        VARIATIONS[v]["Z_edges"] = METISSE_Z_EDGES
        VARIATIONS[v]["Z_grid"] = METISSE_Z_GRID

KNOWN_BHS = {
    "isolated": [
        {
            "name": "OGLE-2011-BLG-0462",
            "mass": 7.15,
            "mass_error": 0.83,
        }
    ],
    "bh-stars": [
        {
            "name": "Gaia BH1",
            "mass": 9.27,
            "mass_error": 0.1,
            "period": 185.387,
            "zmax": 0.262,
            "zmax_error": 0.003
        },
        {
            "name": "Gaia BH2",
            "mass": 8.94,
            "mass_error": 0.34,
            "period": 1276.7,
            "zmax": 0.089,
            "zmax_error": 0.003
        },
        {
            "name": "Gaia BH3",
            "mass": 33.08,
            "mass_error": 0.48,
            "period": 11.57 * 365.25,
            "period_error": 0.25 * 365.25,
            "zmax": 6.785,
            "zmax_error": 0.415
        }
    ]
}