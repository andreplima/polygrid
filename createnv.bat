conda update -n root -c defaults conda
conda remove --name recsys23 --all
python -m pip install --upgrade pip
conda create --name recsys23 shapely scikit-learn=1.2.2 matplotlib=3.3.2 ffmpeg
activate recsys23
conda install psutil
conda install -c conda-forge scikit-fuzzy pyreadstat
python -m pip install bootstrapped
python -m pip install palmerpenguins
python -m pip install scikit-multilearn-ng
python -m pip install primefac

rem # packages in environment at C:\Users\andre\Miniconda3\envs\recsys23:
rem #
rem appdirs                   1.4.4              pyhd3eb1b0_0  
rem blas                      1.0                         mkl  
rem bootstrapped              0.0.2                     <pip>
rem brotlipy                  0.7.0           py39h2bbff1b_1003  
rem ca-certificates           2023.01.10           haa95532_0  
rem certifi                   2022.12.7        py39haa95532_0  
rem cffi                      1.15.1           py39h2bbff1b_3  
rem charset-normalizer        2.0.4              pyhd3eb1b0_0  
rem cryptography              39.0.1           py39h21b164f_0  
rem cycler                    0.11.0             pyhd3eb1b0_0  
rem ffmpeg                    4.2.2                he774522_0  
rem freetype                  2.12.1               ha860e81_0  
rem geos                      3.8.0                h33f27b4_0  
rem giflib                    5.2.1                h8cc25b3_3  
rem glib                      2.69.1               h5dc1a3c_2  
rem gst-plugins-base          1.18.5               h9e645db_0  
rem gstreamer                 1.18.5               hd78058f_0  
rem icc_rt                    2022.1.0             h6049295_2  
rem icu                       58.2                 ha925a31_3  
rem idna                      3.4              py39haa95532_0  
rem intel-openmp              2021.4.0          haa95532_3556  
rem joblib                    1.1.1            py39haa95532_0  
rem jpeg                      9e                   h2bbff1b_1  
rem kiwisolver                1.4.4            py39hd77b12b_0  
rem krb5                      1.19.4               h5b6d351_0  
rem lerc                      3.0                  hd77b12b_0  
rem libclang                  14.0.6          default_hf4acfee_0  
rem libclang13                14.0.6          default_h8992b79_0  
rem libdeflate                1.17                 h2bbff1b_0  
rem libffi                    3.4.2                hd77b12b_6  
rem libiconv                  1.17                 h8ffe710_0    conda-forge
rem libogg                    1.3.5                h2bbff1b_1  
rem libpng                    1.6.39               h8cc25b3_0  
rem libtiff                   4.5.0                h6c2663c_2  
rem libvorbis                 1.3.7                he774522_0  
rem libwebp                   1.2.4                hbc33d0d_1  
rem libwebp-base              1.2.4                h2bbff1b_1  
rem libxml2                   2.9.14               h0ad7f3c_0  
rem libxslt                   1.1.35               h2bbff1b_0  
rem libzlib                   1.2.13               hcfcfb64_4    conda-forge
rem lz4-c                     1.9.4                h2bbff1b_0  
rem matplotlib                3.3.2                haa95532_0  
rem matplotlib-base           3.3.2            py39h35d3fe4_0  
rem mkl                       2021.4.0           haa95532_640  
rem mkl-service               2.4.0            py39h2bbff1b_0  
rem mkl_fft                   1.3.1            py39h277e83a_0  
rem mkl_random                1.2.2            py39hf11a4ad_0  
rem networkx                  2.8.8              pyhd8ed1ab_0    conda-forge
rem numpy                     1.23.5           py39h3b20f71_0  
rem numpy-base                1.23.5           py39h4da318b_0  
rem openssl                   1.1.1t               h2bbff1b_0  
rem packaging                 23.0             py39haa95532_0  
rem pandas                    1.5.1            py39h2ba5b7c_1    conda-forge
rem pcre                      8.45                 hd77b12b_0  
rem pillow                    9.4.0            py39hd77b12b_0  
rem pip                       23.0.1           py39haa95532_0  
rem ply                       3.11             py39haa95532_0  
rem pooch                     1.4.0              pyhd3eb1b0_0  
rem pycparser                 2.21               pyhd3eb1b0_0  
rem pyopenssl                 23.0.0           py39haa95532_0  
rem pyparsing                 3.0.9            py39haa95532_0  
rem pyqt                      5.15.7           py39hd77b12b_0  
rem pyqt5-sip                 12.11.0          py39hd77b12b_0  
rem pyreadstat                1.2.0            py39h585ecb4_1    conda-forge
rem pysocks                   1.7.1            py39haa95532_0  
rem python                    3.9.16               h6244533_2  
rem python-dateutil           2.8.2              pyhd3eb1b0_0  
rem python_abi                3.9                      2_cp39    conda-forge
rem pytz                      2022.6             pyhd8ed1ab_0    conda-forge
rem qt-main                   5.15.2               he8e5bd7_8  
rem qt-webengine              5.15.9               hb9a9bb5_5  
rem qtwebkit                  5.212                h3ad3cdb_4  
rem requests                  2.28.1           py39haa95532_1  
rem scikit-fuzzy              0.4.2                      py_1    conda-forge
rem scikit-learn              1.2.2            py39hd77b12b_0  
rem scipy                     1.10.1           py39h321e85e_0  
rem setuptools                65.6.3           py39haa95532_0  
rem shapely                   2.0.1            py39hd7f5953_0  
rem sip                       6.6.2            py39hd77b12b_0  
rem six                       1.16.0             pyhd3eb1b0_1  
rem sqlite                    3.41.1               h2bbff1b_0  
rem threadpoolctl             2.2.0              pyh0d69192_0  
rem tk                        8.6.12               h2bbff1b_0  
rem toml                      0.10.2             pyhd3eb1b0_0  
rem tornado                   6.2              py39h2bbff1b_0  
rem tzdata                    2023c                h04d1e81_0  
rem ucrt                      10.0.22621.0         h57928b3_0    conda-forge
rem urllib3                   1.26.15          py39haa95532_0  
rem vc                        14.2                 h21ff451_1  
rem vs2015_runtime            14.32.31332          h1d6e394_9    conda-forge
rem wheel                     0.38.4           py39haa95532_0  
rem win_inet_pton             1.1.0            py39haa95532_0  
rem wincertstore              0.2              py39haa95532_2  
rem xz                        5.2.10               h8cc25b3_1  
rem zlib                      1.2.13               h8cc25b3_0  
rem zstd                      1.5.4                hd43e919_0  

