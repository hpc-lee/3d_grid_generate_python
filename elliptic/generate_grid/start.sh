#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJDIR="${SCRIPT_DIR}/output"

rm -rf "${PROJDIR}"
mkdir -p "${PROJDIR}"

INPUTDIR="${SCRIPT_DIR}/../prep_grid_bdry"
NUMPROCS=1

cat << ieof > ${PROJDIR}/config.json
{
    "number_of_grid_points_x" : 51,
    "number_of_grid_points_y" : 41,
    "number_of_grid_points_z" : 21,

    "number_of_mpiprocs_x" : 1,
    "number_of_mpiprocs_y" : 1,
    "number_of_mpiprocs_z" : 1,

    "grid_check" : 1,
    "check_orth" : 1,
    "check_jac"  : 1,
    "check_smooth_xi" : 1,
    "check_smooth_et" : 1,
    "check_smooth_zt" : 1,

    "geometry_input_file" : "${INPUTDIR}/bdry_file_3d.txt",
    "grid_export_dir" : "${PROJDIR}",

    "grid_method" : {
        "elli_diri" : {
            "coef" : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "iter_err" : 1e-4,
            "max_iter" : 500
        }
    }
}
ieof

echo "+ created config.json"

mpiexec -np ${NUMPROCS} python "${SCRIPT_DIR}/ellip.py" \
    --config-file "${PROJDIR}/config.json" \
    --verbose 10 2>&1 | tee "${PROJDIR}/output.log"
