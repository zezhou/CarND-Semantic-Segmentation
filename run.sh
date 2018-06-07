#echo "sudo docker run -it -v $PWD:/work tensorflow/tensorflow:latest-gpu /bin/bash"
#sudo docker run -it -v $PWD:/work tensorflow/tensorflow:latest-gpu /bin/bash
echo "nvidia-docker run -it -p 8888:8888 -v $PWD:/work tensorflow/tensorflow:latest-gpu /bin/bash"
nvidia-docker run -it -p 8888:8888 -v $PWD:/work tensorflow/tensorflow:latest-gpu /bin/bash
#nvidia-docker start 2b44eac1aba5
#nvidia-docker attach 2b44eac1aba5
