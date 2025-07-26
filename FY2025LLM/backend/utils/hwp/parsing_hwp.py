# from config import conf
# import os
#
# # 실행 파일 경로
# current_file_path = os.path.abspath(__file__)
# CURRENT_DIR = os.path.dirname(current_file_path)
# print(CURRENT_DIR)
#
# ############ HWP (구버전) ############################################################
# from pyhwp import hwp5
# from pyhwp.hwp5.bodytext import iter_paragraph_text
# test_hwp_name = "테스트01.hwp"
# test_hwp_path = f"{CURRENT_DIR}/data/{test_pdf_name}"
# with open(test_hwp_name, "rb") as hwp_file:
#     hwp = hwp5.HWP5File(hwp_file)
#     hwp.open()
#
#     text = ""
#     for section in hwp.bodytext.section_list:
#         for para in iter_paragraph_text(section.text):
#             text += para + "\n"
#
# print(text)

# ((venvLinuxPy312) ) devbear@BOOK-MB2VJ96366:/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM$ 
# PYTHONPATH=. python3 utilities/elastic/parsing_hwp.py
# rm -rf /mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM/venvLinuxPy312