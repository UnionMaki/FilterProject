#include "TestingMetods.hpp"

// Rolling Guidance Filter (фильтр сглаживания текстур)
void TestingMetods::rollingGuidanceProcessFrame(ProcessedVideo* obj, cv::Mat frame) {
    if (frame.empty() || obj == nullptr) return;

    cv::Mat result;
    cv::ximgproc::rollingGuidanceFilter(frame, result, 3, 25, 4);

    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = result.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "Rolling Guidance";
}

// Weighted Median Filter (взвешенный медианный фильтр)
void TestingMetods::weightedMedianFilterFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat result;
    // используем ту же матрицу как веса
    cv::ximgproc::weightedMedianFilter(frame, frame, result, 5, 0.5);

    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = result.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "Weighted Median";
}

// NiBlack Threshold (адаптивная бинаризация)
void TestingMetods::niBlackThresholdFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat gray, result;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    // метод: 0 = BINARIZATION_NIBLACK, 1 = BINARIZATION_SAUVOLA
    cv::ximgproc::niBlackThreshold(gray, result, 255, cv::THRESH_BINARY, 15, 0.2, 0);

    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = result.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "Black Threshold";
}

// Anisotropic Diffusion (анизотропная диффузия)
void TestingMetods::anisotropicDiffusionFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat result;
    // 10 итераций, alpha = 0.1, k = 0.01
    cv::ximgproc::anisotropicDiffusion(frame, result, 10, 0.1, 0.01);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = result.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "Anisotropic Diffusion";
}

void TestingMetods::bilateralSharpprocessFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage, sharpenedImage;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75);
    cv::filter2D(blurredImage, sharpenedImage, blurredImage.depth(), kernel);
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "bilateral_sharpened";
}

void TestingMetods::GrayGaussprocessFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat grayImage;
    cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);
    VideoFrame Fr;
    Fr.original = grayImage.clone();
    Fr.processed = blurredImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "Grey + Gaussian";
}

void TestingMetods::GaussprocessFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage;
    cv::GaussianBlur(frame, blurredImage, cv::Size(5, 5), 0);
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = blurredImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "Gaussian";
}

void TestingMetods::MedianprocessFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage;
    cv::medianBlur(frame, blurredImage, 5);
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = blurredImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "Median";
}

void TestingMetods::bilateralprocessFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage;
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75); // хуй знает что делают аргументы
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = blurredImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "bilateral";
}

void TestingMetods::fastNlprocessFrame(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage;
    cv::fastNlMeansDenoisingColored(frame, blurredImage, 10, 10, 21); // хуй знает что делают аргументы
    // у этой функции кстати есть улучшение связанное с обработкой нескольких кадров сразу, может быть полезно
    // ссылка на документацию https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = blurredImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "fastNlMeans";
}

// 1. ГРУБАЯ РЕЗКОСТЬ (Сильный перепад, эффект "перешарпа")
// Разница: Очень темные и яркие ореолы на границах, агрессивный вид.
void TestingMetods::bilateralSharpHeavy(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage, sharpenedImage;
    // Сумма коэффициентов = 1. Сильное вычитание соседей.
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, 
                                                -1,  9, -1, 
                                                -1, -1, -1);
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75);
    cv::filter2D(blurredImage, sharpenedImage, blurredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "bilateral_sharp_heavy";
}

// 2. МЯГКАЯ РЕЗКОСТЬ (Unsharp Mask аналог)
// Разница: Почти незаметно на первый взгляд, но убирает "мыло" с билатерала, не создавая ореолов.
void TestingMetods::bilateralSharpSubtle(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage, sharpenedImage;
    // Сумма коэффициентов = 1. Очень слабое влияние.
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -0.1, -0.1, -0.1, 
                                                -0.1,  1.4, -0.1, 
                                                -0.1, -0.1, -0.1);
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75);
    cv::filter2D(blurredImage, sharpenedImage, blurredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "bilateral_sharp_subtle";
}

// 3. ГОРИЗОНТАЛЬНАЯ/ВЕРТИКАЛЬНАЯ РЕЗКОСТЬ (Эффект "дрожания" или выделение прямых линий)
// Разница: Резкость проявляется ТОЛЬКО на вертикальных и горизонтальных границах. Диагонали остаются мягкими.
void TestingMetods::bilateralSharpCross(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage, sharpenedImage;
    // Усиливаем только центр и крест, углы не трогаем.
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -0.5, 0, 
                                                -0.5, 3.0, -0.5, 
                                                0, -0.5, 0);
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75);
    cv::filter2D(blurredImage, sharpenedImage, blurredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "bilateral_sharp_cross";
}

// 4. ОБЪЕМНАЯ РЕЗКОСТЬ (Emboss / Тиснение)
// Разница: Изображение становится "серым" с яркими краями, как барельеф. Кардинально другой визуал.
void TestingMetods::bilateralSharpEmboss(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage, sharpenedImage;
    // Сумма коэффициентов = 0. Это даст смещение яркости в серый + контур.
    // Для приведения к норме нужно добавить смещение (delta). filter2D поддерживает delta.
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, 0, 
                                                -1,  0, 1, 
                                                0,  1, 1);
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75);
    // Добавляем delta = 128, чтобы вернуть яркость в середину диапазона
    cv::filter2D(blurredImage, sharpenedImage, blurredImage.depth(), kernel, cv::Point(-1, -1), 128.0);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "bilateral_sharp_emboss";
}

// 5. РЕЗКОСТЬ ТОЛЬКО ПО КРАСНОМУ КАНАЛУ (Хроматическая аберрация)
// Разница: Разница с оригиналом огромная — появляется цветной ореол (красный/синий) на границах объектов.
void TestingMetods::bilateralSharpChromatic(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage, sharpenedImage;
    // Стандартное ядро резкости
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75);
    
    // Разделяем каналы
    std::vector<cv::Mat> channels(3);
    cv::split(blurredImage, channels);
    
    // Применяем резкость ТОЛЬКО к красному каналу, а к синему — инвертированную матрицу (размытие)
    cv::filter2D(channels[2], channels[2], blurredImage.depth(), kernel); // R channel sharpened
    
    cv::Mat kernelBlur = (cv::Mat_<float>(3, 3) << 0, 0.1, 0, 0.1, 0.6, 0.1, 0, 0.1, 0);
    cv::filter2D(channels[0], channels[0], blurredImage.depth(), kernelBlur); // B channel slightly blurred
    
    cv::merge(channels, sharpenedImage);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "bilateral_sharp_chromatic";
}

// 6. КОЛЬЦЕВАЯ РЕЗКОСТЬ (Эффект радиального контура)
// Разница: Выделяет объекты круглым ореолом, похоже на неудачную работу AI-апскейла.
void TestingMetods::bilateralSharpRing(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat blurredImage, sharpenedImage;
    // Матрица, где от центра отнимаются соседи по кругу (радиус 2 в пределах 3х3)
    // Выглядит очень специфично
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -0.5, -1.0, -0.5, 
                                                -1.0,  7.0, -1.0, 
                                                -0.5, -1.0, -0.5);
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75);
    cv::filter2D(blurredImage, sharpenedImage, blurredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "bilateral_sharp_ring";
}

// 1. Guided Filter + ТЯЖЕЛАЯ РЕЗКОСТЬ
// Разница: Guided Filter лучше сохраняет края, чем Bilateral, поэтому резкость ложится чище, без ореолов на перепадах яркости.
void TestingMetods::guidedSharpHeavy(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat filteredImage, sharpenedImage;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, 
                                                -1,  9, -1, 
                                                -1, -1, -1);
    // Guided Filter: radius=8, eps=0.01^2 * 255^2 ≈ 6.5
    cv::ximgproc::guidedFilter(frame, frame, filteredImage, 8, 6.5);
    cv::filter2D(filteredImage, sharpenedImage, filteredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "guided_sharp_heavy";
}

// 2. Rolling Guidance Filter + МЯГКАЯ РЕЗКОСТЬ
// Разница: Убирает мелкие детали (текстуру кожи, шум), но оставляет крупные границы. Резкость применяется только к "важным" контурам.
void TestingMetods::rollingSharpSubtle(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat filteredImage, sharpenedImage;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -0.1, -0.1, -0.1, 
                                                -0.1,  1.4, -0.1, 
                                                -0.1, -0.1, -0.1);
    
    // Rolling Guidance: убирает текстуру масштаба меньше 5 пикселей
    cv::ximgproc::rollingGuidanceFilter(frame, filteredImage, 5, 0.05, 4);
    cv::filter2D(filteredImage, sharpenedImage, filteredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "rolling_sharp_subtle";
}

// 3. Anisotropic Diffusion (Perona-Malik) + ПЕРЕКРЕСТНАЯ РЕЗКОСТЬ
// Разница: Фильтр "течет" вдоль границ, создавая эффект акварели или масляной живописи. Резкость подчеркивает направление мазков.
void TestingMetods::anisotropicSharpCross(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat filteredImage, sharpenedImage, gray, flow;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -0.5, 0, 
                                                -0.5, 3.0, -0.5, 
                                                0, -0.5, 0);
    
    // Переводим в float для anisotropic diffusion
    cv::Mat frameFloat;
    frame.convertTo(frameFloat, CV_32FC3, 1.0/255.0);
    
    // Anisotropic Diffusion (K=0.02, iterations=10)
    cv::ximgproc::anisotropicDiffusion(frameFloat, filteredImage, 0.02f, 0.1f, 10);
    
    filteredImage.convertTo(filteredImage, CV_8UC3, 255.0);
    cv::filter2D(filteredImage, sharpenedImage, filteredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "anisotropic_sharp_cross";
}

// 4. L0 Smoothing + ЭМБОСС
// Разница: L0 выравнивает регионы в плоские цветовые пятна, сохраняя ТОЛЬКО сильные границы. Эмбосс делает из этого настоящий барельеф.
void TestingMetods::l0SmoothSharpEmboss(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat filteredImage, sharpenedImage;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, 0, 
                                                -1,  0, 1, 
                                                0,  1, 1);
    
    // L0 Smoothing: lambda=0.02
    cv::ximgproc::l0Smooth(frame, filteredImage, 0.02);
    
    cv::filter2D(filteredImage, sharpenedImage, filteredImage.depth(), kernel, cv::Point(-1, -1), 128.0);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "l0_smooth_sharp_emboss";
}

// 5. Fast Global Smoother + ХРОМАТИЧЕСКАЯ РЕЗКОСТЬ
// Разница: FGS создает очень гладкие градиенты, на которых цветной ореол от хроматической резкости выглядит как 3D-аберрация.
void TestingMetods::fgsSharpChromatic(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat filteredImage, sharpenedImage;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    
    // Fast Global Smoother (работает поканально)
    std::vector<cv::Mat> channels(3);
    cv::split(frame, channels);
    
    cv::Ptr<cv::ximgproc::FastGlobalSmootherFilter> fgs = 
        cv::ximgproc::createFastGlobalSmootherFilter(frame, 0.1, 100.0);
    
    for(int i = 0; i < 3; i++) {
        cv::Mat guide = channels[i];
        fgs->filter(guide, channels[i]);
    }
    cv::merge(channels, filteredImage);
    
    // Применяем резкость только к R и B каналам с разными знаками
    std::vector<cv::Mat> sharpChannels(3);
    cv::split(filteredImage, sharpChannels);
    
    cv::filter2D(sharpChannels[2], sharpChannels[2], filteredImage.depth(), kernel);      // R +
    cv::Mat kernelInv = -kernel;
    cv::filter2D(sharpChannels[0], sharpChannels[0], filteredImage.depth(), kernelInv);   // B -
    
    cv::merge(sharpChannels, sharpenedImage);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "fgs_sharp_chromatic";
}

// 6. Edge-Preserving Decomposition (DT Filter) + КОЛЬЦЕВАЯ РЕЗКОСТЬ
// Разница: Domain Transform фильтр создает ступенчатые границы. Кольцевая резкость на них дает эффект "мультяшной" обводки.
void TestingMetods::dtSharpRing(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat filteredImage, sharpenedImage;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -0.5, -1.0, -0.5, 
                                                -1.0,  7.0, -1.0, 
                                                -0.5, -1.0, -0.5);
    
    // Domain Transform Filter (sigmaSpatial=50, sigmaColor=0.1)
    cv::ximgproc::dtFilter(frame, frame, filteredImage, 50.0, 0.1);
    
    cv::filter2D(filteredImage, sharpenedImage, filteredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "dt_sharp_ring";
}

// 7. Weighted Median Filter + ЭКСТРЕМАЛЬНАЯ РЕЗКОСТЬ (ДИАМАНТ)
// Разница: WMF убирает шум и оставляет идеально ровные границы. Резкость получается "цифровой", без артефактов, но очень жесткой.
void TestingMetods::wmfSharpDiamond(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat filteredImage, sharpenedImage;
    // Крупное ядро резкости (диамант)
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -2, -1, -2, 
                                                -1, 13, -1, 
                                                -2, -1, -2);
    
    // Weighted Median Filter (r=5)
    cv::ximgproc::weightedMedianFilter(frame, frame, filteredImage, 5);
    
    cv::filter2D(filteredImage, sharpenedImage, filteredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "wmf_sharp_diamond";
}

// 8. Joint Bilateral Filter + ДАЛЬНЯЯ РЕЗКОСТЬ (Large Kernel)
// Разница: Использует guide image для фильтрации. С дальним радиусом резкости (5x5) создает эффект объемной фотографии.
void TestingMetods::jointBilateralSharpLarge(ProcessedVideo* obj, cv::Mat frame) {
    cv::Mat filteredImage, sharpenedImage;
    
    // Ядро 5x5 для глубокой резкости (High Pass Overshoot)
    cv::Mat kernel = (cv::Mat_<float>(5, 5) << 
        0,  0, -1,  0,  0,
        0, -1, -2, -1,  0,
    -1, -2, 21, -2, -1,
        0, -1, -2, -1,  0,
        0,  0, -1,  0,  0);
    
    // Joint Bilateral Filter: используем само изображение как guide
    cv::ximgproc::jointBilateralFilter(frame, frame, filteredImage, 7, 25.0, 25.0);
    
    cv::filter2D(filteredImage, sharpenedImage, filteredImage.depth(), kernel);
    
    VideoFrame Fr;
    Fr.original = frame.clone();
    Fr.processed = sharpenedImage.clone();
    obj->Framelist.push_back(Fr);
    obj->process_type = "joint_bilateral_sharp_large";
}