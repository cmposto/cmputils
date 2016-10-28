cmake_minimum_required(VERSION 3.5)

find_program(PYTHON "python")

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmputils_module.pyx ${CMAKE_CURRENT_BINARY_DIR}/cmputils_module.pyx)

    add_custom_command(OUTPUT ${OUTPUT}
            COMMENT "Run Cython"
            COMMAND ${PYTHON} ${SETUP_PY} build_ext -i
            COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
            DEPENDS ${TEST_PROJECT_NAME})

    add_custom_target(cython ALL DEPENDS ${OUTPUT})

    #install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif(PYTHON)