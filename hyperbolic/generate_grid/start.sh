#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJDIR="${SCRIPT_DIR}/output"

rm -rf "${PROJDIR}"
mkdir -p "${PROJDIR}"

INPUTDIR="${SCRIPT_DIR}/../prep_grid_bdry"

cat << ieof > ${PROJDIR}/config.json
{
    "number_of_grid_points_x" : 51,
    "number_of_grid_points_y" : 41,
    "number_of_grid_points_z" : 21,

    "check_orth" : 1,
    "check_jac"  : 1,
    "check_step_xi" : 0,
    "check_step_et" : 0,
    "check_step_zt" : 0,
    "check_smooth_xi" : 1,
    "check_smooth_et" : 1,
    "check_smooth_zt" : 1,

    "geometry_input_file" : "${INPUTDIR}/data_file_3d.txt",
    "grid_export_dir" : "${PROJDIR}",

    "flag_stretch" : 0,

    "hyperbolic" : {
        "coef" : 65.0,
        "bdry_x_type" : 1,
        "epsilon_x" : 0.0,
        "bdry_y_type" : 1,
        "epsilon_y" : 0.0,
        "direction" : "z",
        "t2b" : 1,
        "step_input_file" : "${INPUTDIR}/step_file_3d.txt"
    }
}
ieof

echo "+ created config.json"

python "${SCRIPT_DIR}/hyper.py" \
    --config-file "${PROJDIR}/config.json" \
    --verbose 10 2>&1 | tee "${PROJDIR}/output.log"
