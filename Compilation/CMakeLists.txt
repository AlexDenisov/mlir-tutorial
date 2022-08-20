macro(compile_ruby input)
  set (bc_output ${CMAKE_BINARY_DIR}/${input}.bc)
  add_custom_command(
    OUTPUT ${bc_output}
    COMMAND $<TARGET_FILE:ruby_mlir_tutorial> --output=${bc_output}
  )
  add_custom_target(compile-${input}
    DEPENDS ${bc_output}
  )
  add_executable(${input}.exe
    runtime.c
  )
#  add_dependencies(${input}.exe compile-${input})
  target_link_libraries(${input}.exe PRIVATE
    mruby
    mruby_core
    ${bc_output}
  )
  target_link_directories(${input}.exe PRIVATE ${MRUBY_DIR}/build/host/lib)
  target_include_directories(${input}.exe PRIVATE
    ${MRUBY_DIR}/include
    )
endmacro()

compile_ruby(hello)
