all:
	g++ src/*.cpp -o main -I./inc/ -I./include/ -L./lib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_imgcodecs -lopencv_imgproc -lopencv_xfeatures2d 

.PHONY:clean
clean:
	@rm -f ./image/test_*.jpg ./image/result.jpg main