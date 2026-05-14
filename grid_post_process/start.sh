#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJDIR="${SCRIPT_DIR}/output"

rm -rf "${PROJDIR}"
mkdir -p "${PROJDIR}"

INPUTDIR="${SCRIPT_DIR}/../parabolic/generate_grid/output"

cat << ieof > ${PROJDIR}/config.json
{
    "input_grid_number" : 1,

    "input_grid_info_0" : {
        "grid_import_dir" : "${INPUTDIR}",
        "number_of_grid_points" : [51, 41, 21],
        "number_of_mpiprocs_in" : [1, 2, 1],
        "flag_stretch" : 0,
        "stretch_direction" : "z",
        "stretch_file" : ""
    },

    "number_of_mpiprocs_out" : [1, 1, 1],

    "grid_check" : 1,
    "check_orth" : 1,
    "check_jac"  : 1,
    "check_smooth_xi" : 1,
    "check_smooth_et" : 1,
    "check_smooth_zt" : 1,

    "flag_sample" : 0,
    "sample_factor" : [2, 2, 2],

    "flag_min_dist" : 0,

    "pml_weight_2x" : 0,
    "number_of_pml_x1" : 0,
    "number_of_pml_x2" : 0,
    "number_of_pml_y1" : 0,
    "number_of_pml_y2" : 0,
    "number_of_pml_z1" : 0,
    "number_of_pml_z2" : 0,

    "grid_export_dir" : "${PROJDIR}"
}
ieof

echo "+ created config.json"

python "${SCRIPT_DIR}/post_pro.py" \
    --config-file "${PROJDIR}/config.json" \
    --verbose 10 2>&1 | tee "${PROJDIR}/output.log"
