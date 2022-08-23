add_executable(hello-mruby main.c)
add_dependencies(hello-mruby MRuby)
target_link_libraries(hello-mruby PRIVATE
  mruby
  mruby_core
  )
target_link_directories(hello-mruby PRIVATE ${MRUBY_DIR}/build/host/lib)
target_include_directories(hello-mruby PRIVATE
  ${MRUBY_DIR}/include
  )
