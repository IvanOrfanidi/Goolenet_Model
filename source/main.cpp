#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

constexpr std::string_view NAME_LABEL_FILE = "synset_words.txt"; // Tag file.
constexpr std::string_view NAME_DEPLOY_FILE = "bvlc_googlenet.prototxt"; // Description file.
constexpr std::string_view NAME_MODEL_FILE = "bvlc_googlenet.caffemodel"; // Training files.

constexpr std::string_view OUTPUT_NAME_FILE = "output.avi";

constexpr int WIDTH = 500;
constexpr int HEIGHT = 500;
constexpr int DELAY_MS = 100;

void getLabelsFromFile(std::vector<std::string>& labels, const std::string& nameFile)
{
    std::ifstream file;
    file.open(nameFile, std::ifstream::in);
    if (file.is_open()) {
        while (!file.eof()) {
            std::string line;
            std::getline(file, line);
            const auto name = line.substr(line.find(" ") + 1);
            labels.push_back(name);
        }

        file.close();
    }
}

int main()
{
    // Open the default video camera.
    cv::VideoCapture capture(cv::VideoCaptureAPIs::CAP_ANY);
    if (capture.isOpened() == false) {
        std::cerr << "Cannot open the video camera!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string path = std::filesystem::current_path().string() + '/';
    std::replace(path.begin(), path.end(), '\\', '/');

    const auto width = capture.get(cv::CAP_PROP_FRAME_WIDTH); // Get the width of frames of the video.
    const auto height = capture.get(cv::CAP_PROP_FRAME_HEIGHT); // Get the height of frames of the video.
    const auto fps = capture.get(cv::CAP_PROP_FPS);
    std::cout << "Resolution of the video: " << width << " x " << height << ".\nFrames per seconds: " << fps << "." << std::endl;

    std::vector<std::string> labels;
    getLabelsFromFile(labels, path + NAME_LABEL_FILE.data());
    if (labels.empty()) {
        std::cerr << "Failed to read file!" << std::endl;
        return EXIT_FAILURE;
    }

    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    cv::VideoWriter video(OUTPUT_NAME_FILE.data(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(WIDTH, HEIGHT));

    cv::Mat source;
    static constexpr int ESCAPE_KEY = 27;
    while (cv::waitKey(DELAY_MS) != ESCAPE_KEY) {
        // Read a new frame from video.
        if (capture.read(source) == false) { // Breaking the while loop if the frames cannot be captured.
            std::cerr << "Video camera is disconnected!" << std::endl;
            return EXIT_FAILURE;
        }

        cv::dnn::Net net;
        // Read binary file and description file.
        net = cv::dnn::readNetFromCaffe(path + NAME_DEPLOY_FILE.data(), path + NAME_MODEL_FILE.data());
        if (net.empty()) {
            std::cerr << "Could not load Caffe_net!" << std::endl;
            return EXIT_FAILURE;
        }

        const auto start = cv::getTickCount();

        // Convert the read image to blob.
        static constexpr double scalefactor = 1.0; // The image pixel value is scaled by the scaling factor (Scalefactor).
        const cv::Mat blob = cv::dnn::blobFromImage(source, // Input the image to be processed or classified by the neural network.
            scalefactor, // After the image is subtracted from the average value, the remaining pixel values ​​are scaled to a certain extent.
            cv::Size(224, 224), // Neural network requires the input image size during training.
            cv::Scalar(104, 117, 123) /* Mean needs to subtract the average value of the image as a whole.
                                      If we need to subtract different values ​​from the three channels of the RGB image,
                                      then 3 sets of average values ​​can be used. */
        );

        net.setInput(blob, "data");
        const cv::Mat score = net.forward("prob");
        std::string runTime = "run time: " + std::to_string(static_cast<double>(cv::getTickCount() - start) / cv::getTickFrequency());
        runTime.erase(runTime.end() - 3, runTime.end());

        const cv::Mat mat = score.reshape(1, 1); // The dimension becomes 1*1000.
        double probability; // Maximum similarity.
        cv::Point index;
        cv::minMaxLoc(mat, nullptr, &probability, nullptr, &index);

        resize(source, source, cv::Size(WIDTH, HEIGHT), 0, 0);

        const auto& name = labels.at(static_cast<size_t>(index.x)); // Index corresponding to maximum similarity.
        cv::putText(source, name, cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 1, 5);

        cv::putText(source, runTime, cv::Point(10, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);
#ifdef NDEBUG
        cv::putText(source, "in release", cv::Point(150, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);
#else
        cv::putText(source, "in debug", cv::Point(150, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);
#endif
        cv::putText(source, "probability: " + std::to_string(int(probability * 100)) + "%", cv::Point(260, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);
        const std::string resolution = std::to_string(source.size().width) + "x" + std::to_string(source.size().height);
        cv::putText(source, resolution, cv::Point(source.size().width - 80, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);

        cv::imshow("FCN-demo", source);

        // Write the frame into the file.
        video.write(source);
    }

    capture.release();
    video.release();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
