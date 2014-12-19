a5cuda
======

A5/1 chain generator implemented in CUDA.

Credit: my credit to Karsten Nohl, Frank A. Stevenson for their work in Cpu and Ati Radeon platform, which i based on to build this version

1.3 release note:
Since there is most likely no support for std::thread in CUDA 6.5, so I decided to comeback with Boost thread.
- switch std::thread to boost::thread
- dynamically specify blocksize and gridsize

1.2

1.1

1.0
