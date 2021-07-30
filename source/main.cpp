#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include <boost/program_options.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

constexpr std::string_view NAME_LABEL_FILE = "synset_words.txt"; // Tag file.
constexpr std::string_view NAME_DEPLOY_FILE = "bvlc_googlenet.prototxt"; // Description file.
constexpr std::string_view NAME_MODEL_FILE = "bvlc_googlenet.caffemodel"; // Training files.

constexpr int WIDTH = 500;
constexpr int HEIGHT = 500;
constexpr int DELAY_MS = 1;

void getLabelsFromFile(std::vector<std::string>& labels, const std::string& nameFile)
{
    std::ifstream file;
    file.open(nameFile, std::ifstream::in);
    if (file.is_open()) {
        while (!file.eof()) {
            std::string line;
            std::getline(file, line);
            const std::string name = line.substr(line.find(" ") + 1);
            labels.push_back(std::move(name));
        }

        file.close();
    }
}

int main(int argc, char* argv[])
{
    std::string inputFile;
    std::string outputFile;
    bool useCuda;
    uint16_t frameNumber;
    boost::program_options::options_description desc("Options");
    desc.add_options()
        // All options:
        ("in,i", boost::program_options::value<std::string>(&inputFile)->default_value(""), "Path to input file.\n") //
        ("out,o", boost::program_options::value<std::string>(&outputFile)->default_value("output.mp4"), "Path to output file.\n") //
        ("cuda,c", boost::program_options::value<bool>(&useCuda)->default_value(true), "Set CUDA Enable.\n") //
        ("frame,f", boost::program_options::value<uint16_t>(&frameNumber)->default_value(1), "Set frame number.") //
        ("help,h", "Produce help message."); // Help
    boost::program_options::variables_map options;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), options);
        boost::program_options::notify(options);
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
    if (options.count("help")) {
        std::cout << desc << std::endl;
        return EXIT_SUCCESS;
    }

    cv::VideoCapture capture;
    if (inputFile.length() == 0) {
        // Open default video camera
        capture.open(cv::VideoCaptureAPIs::CAP_ANY);
    } else {
        capture.open(inputFile);
    }
    if (capture.isOpened() == false) {
        std::cerr << "Cannot open video!" << std::endl;
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
    cv::VideoWriter video(outputFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(WIDTH, HEIGHT));

    bool cudaEnable = false;
    if (cv::cuda::getCudaEnabledDeviceCount() != 0) {
        cv::cuda::DeviceInfo deviceInfo;
        if (deviceInfo.isCompatible() != 0 && useCuda) {
            cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
            cudaEnable = true;
        }
    }

    static constexpr int ESCAPE_KEY = 27;
    while (cv::waitKey(DELAY_MS) != ESCAPE_KEY) {
        // Read a new frame from video.
        cv::Mat source;
        uint16_t i = 0;
        do {
            if (capture.read(source) == false) {
                std::cerr << "Video camera is disconnected!" << std::endl;
                return EXIT_FAILURE;
            }
            ++i;
        } while (i < frameNumber);

        cv::dnn::Net neuralNetwork;
        // Read binary file and description file.
        neuralNetwork = cv::dnn::readNetFromCaffe(path + NAME_DEPLOY_FILE.data(), path + NAME_MODEL_FILE.data());
        if (neuralNetwork.empty()) {
            std::cerr << "Could not load Caffe_net!" << std::endl;
            return EXIT_FAILURE;
        }

        // Set CUDA as preferable backend and target
        if (cudaEnable) {
            neuralNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            neuralNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }

        const auto startTime = cv::getTickCount();
        // Convert the read image to blob.
        static constexpr double scalefactor = 1.0; // The image pixel value is scaled by the scaling factor (Scalefactor).
        const cv::Mat blob = cv::dnn::blobFromImage(source, // Input the image to be processed or classified by the neural network.
            scalefactor, // After the image is subtracted from the average value, the remaining pixel values ​​are scaled to a certain extent.
            cv::Size(224, 224), // Neural network requires the input image size during training.
            cv::Scalar(104, 117, 123) /* Mean needs to subtract the average value of the image as a whole.
                                      If we need to subtract different values ​​from the three channels of the RGB image,
                                      then 3 sets of average values ​​can be used. */
        );

        neuralNetwork.setInput(blob, "data");
        const cv::Mat score = neuralNetwork.forward("prob");
        std::string runTime = "run time: " + std::to_string(static_cast<double>(cv::getTickCount() - startTime) / cv::getTickFrequency());
        runTime.erase(runTime.end() - 3, runTime.end());

        const cv::Mat result = score.reshape(1, 1); // The dimension becomes 1*1000.
        double probability; // Maximum similarity.
        cv::Point index;
        cv::minMaxLoc(result, nullptr, &probability, nullptr, &index);

        resize(source, source, cv::Size(WIDTH, HEIGHT), 0, 0);

        const auto& name = labels.at(static_cast<size_t>(index.x)); // Index corresponding to maximum similarity.
        cv::putText(source, name, cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 1, 5);

        cv::putText(source, runTime, cv::Point(10, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#ifdef NDEBUG
        cv::putText(source, "in release", cv::Point(180, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#else
        cv::putText(source, "in debug", cv::Point(180, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#endif
        if (cudaEnable) {
            cv::putText(source, "using GPUs", cv::Point(300, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
        } else {
            cv::putText(source, "using CPUs", cv::Point(300, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
        }
        const std::string resolution = std::to_string(source.size().width) + "x" + std::to_string(source.size().height);
        cv::putText(source, resolution, cv::Point(source.size().width - 80, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);

        cv::imshow("GooleNet-demo", source);

        // Write the frame into the file.
        video.write(source);
    }

    capture.release();
    video.release();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
