import pandas as pd
# from IPython.display import display
from Indicator.MI import *
# from Indicator.FMI import *
# from Indicator.TE import *
# from Indicator.NCIE import *
from Indicator.QG import *
# from Indicator.QM import *
# from Indicator.SF import *
# from Indicator.structural_similarity import *
# from Indicator.Piella import *
# from Indicator.Cvejie import *
# from Indicator.Yang import *
# from Indicator.MSE import *
from Indicator.no_reference import *
# from Indicator.CE import *
# from Indicator.CC import *
from Indicator.VIF import *

'''
def show_indicator(solution, image1, image2, image_fused):
    image_fused = cv2.resize(image_fused, (image1.shape[1], image1.shape[0]))

    indicators = {
        "MI": calculate_mutual_information_metric(image1, image2, image_fused),
        "NMI": calculate_normalized_mutual_information_metric(image1, image2, image_fused),
        "FMI": fmi(image1, image2, image_fused),
        # "TE": calculate_tsallis_entropy_metric(image1, image2, image_fused),
        "NCIE": calculate_nonlinear_correlation_information_entropy(image1, image2, image_fused),
        "QG": calculate_edge_information_preservation_fused(image1, image2, image_fused), #Q(AB/F)
        "QM": calculate_multiscale_scheme(image1, image2, image_fused), 
        "SF": calculate_spatial_frequency(image_fused),
        "SSIM": calculate_structural_similarity_index_measure_fused(image1, image2, image_fused),
        # "UIQI": calculate_universal_image_quality_index_fused(image1, image2, image_fused),
        "Qs-Piella": calculate_Qs(image1, image2, image_fused),
        "Qw-Piella": calculate_Qw(image1, image2, image_fused),
        # "Qe-Piella": calculate_Qe(image1, image2, image_fused),
        # "Cvejie": calculate_cvejie_metric(image1, image2, image_fused),
        "Yang": calculate_yang_metric(image1, image2, image_fused),
        # "MSE": compute_mean_square_error(image1, image2, image_fused),
        # "RMSE": compute_root_mean_square_error(image1, image2, image_fused),
        "PSNR": compute_peak_signal_to_noise_ratio(image1, image2, image_fused),
        "EN": calculate_entropy(image_fused),
        "AG": calculate_average_gradient(image_fused),
        "SD": calculate_standard_deviation(image_fused),
        # "EI": calculate_edge_intensity(image_fused),
        "ALI": calculate_average_light_intensity(image_fused),
        # "CE": calculate_cross_entropy(image1, image2, image_fused),
        "CC": calculate_cross_correlation(image1, image2, image_fused),
        # "SCD": calculate_sum_of_correlation_difference(image1, image2, image_fused),
        "VIF": calculate_visual_information_fidelity(image1, image2, image_fused),
        # "VIFF": calculate_VIFF(image_fused, image1, image2)
    }

    df = pd.DataFrame(list(indicators.items()), columns=["Phương pháp", solution]).set_index("Phương pháp").T

    # df1 = df.iloc[:, :9]  
    # df2 = df.iloc[:, 9:18]  
    # df3 = df.iloc[:, 18:] 

    # display(df1.head())
    # display(df2.head())
    # display(df3.head())

    return df

def show_indicator_csv(code, image1, image2, image_fused):
        # Resize image_fused to the same size of image1 and image2
        image_fused = cv2.resize(image_fused, (image1.shape[1], image1.shape[0]))

        MI = calculate_mutual_information_metric(image1, image2, image_fused)
        NMI = calculate_normalized_mutual_information_metric(image1, image2, image_fused)
        FMI = fmi(image1, image2, image_fused)
        NCIE = calculate_nonlinear_correlation_information_entropy(image1, image2, image_fused)
        QG = calculate_edge_information_preservation_fused(image1, image2, image_fused) 
        QM = calculate_multiscale_scheme(image1, image2, image_fused)
        SSIM = calculate_structural_similarity_index_measure_fused(image1, image2, image_fused)
        Qs_Piella = calculate_Qs(image1, image2, image_fused)
        Qw_Piella = calculate_Qw(image1, image2, image_fused)
        Qe_Piella = calculate_Qe(image1, image2, image_fused)
        Yang = calculate_yang_metric(image1, image2, image_fused)
        SF = calculate_spatial_frequency(image_fused)
        PSNR = compute_peak_signal_to_noise_ratio(image1, image2, image_fused)
        EN = calculate_entropy(image_fused)
        AG = calculate_average_gradient(image_fused)
        SD = calculate_standard_deviation(image_fused)
        ALI = calculate_average_light_intensity(image_fused)
        CC = calculate_cross_correlation(image1, image2, image_fused)
        VIF = calculate_visual_information_fidelity(image1, image2, image_fused)

        return MI, NMI, FMI, NCIE, QG, QM, SSIM, Qs_Piella, Qw_Piella, Qe_Piella, Yang, SF, PSNR, EN, AG, SD, ALI, CC, VIF

def show_indicator_csv_noFMI(code, image1, image2, image_fused):
        # Resize image_fused to the same size of image1 and image2
        image_fused = cv2.resize(image_fused, (image1.shape[1], image1.shape[0]))

        MI = calculate_mutual_information_metric(image1, image2, image_fused)
        NMI = calculate_normalized_mutual_information_metric(image1, image2, image_fused)
        NCIE = calculate_nonlinear_correlation_information_entropy(image1, image2, image_fused)
        QG = calculate_edge_information_preservation_fused(image1, image2, image_fused) 
        QM = calculate_multiscale_scheme(image1, image2, image_fused)
        SSIM = calculate_structural_similarity_index_measure_fused(image1, image2, image_fused)
        Qs_Piella = calculate_Qs(image1, image2, image_fused)
        Qw_Piella = calculate_Qw(image1, image2, image_fused)
        Qe_Piella = calculate_Qe(image1, image2, image_fused)
        Yang = calculate_yang_metric(image1, image2, image_fused)
        SF = calculate_spatial_frequency(image_fused)
        PSNR = compute_peak_signal_to_noise_ratio(image1, image2, image_fused)
        EN = calculate_entropy(image_fused)
        AG = calculate_average_gradient(image_fused)
        SD = calculate_standard_deviation(image_fused)
        ALI = calculate_average_light_intensity(image_fused)
        CC = calculate_cross_correlation(image1, image2, image_fused)
        VIF = calculate_visual_information_fidelity(image1, image2, image_fused)

        return MI, NMI, NCIE, QG, QM, SSIM, Qs_Piella, Qw_Piella, Qe_Piella, Yang, SF, PSNR, EN, AG, SD, ALI, CC, VIF

'''

def show_indicator_csv_6(code, image1, image2, image_fused):
        # Resize image_fused to the same size of image1 and image2
        image_fused = cv2.resize(image_fused, (image1.shape[1], image1.shape[0]))

        MI = calculate_mutual_information_metric(image1, image2, image_fused)
        QG = calculate_edge_information_preservation_fused(image1, image2, image_fused) 
        EN = calculate_entropy(image_fused)
        SD = calculate_standard_deviation(image_fused)
        ALI = calculate_average_light_intensity(image_fused)
        VIF = calculate_visual_information_fidelity(image1, image2, image_fused)

        return MI, QG, EN, SD, ALI, VIF

def show_indicator_csv_5(code, image1, image2, image_fused):
        # Resize image_fused to the same size of image1 and image2
        image_fused = cv2.resize(image_fused, (image1.shape[1], image1.shape[0]))

        MI = calculate_mutual_information_metric(image1, image2, image_fused)
        QG = calculate_edge_information_preservation_fused(image1, image2, image_fused)
        EN = calculate_entropy(image_fused)
        SD = calculate_standard_deviation(image_fused)
        ALI = calculate_average_light_intensity(image_fused)

        return MI, QG, EN, SD, ALI