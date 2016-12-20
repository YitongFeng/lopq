
file = '../Data/FV/gt_file.txt'

indices = [0, 0, 1, 1, 2, 2]
results_one_img_new = []
results_one_img = [(0, 0.1), (1, 0.2), (2, 0.2)]
results_one_img_new = [indices[rr[0]] for rr in results_one_img if indices[rr[0]] not in results_one_img_new]

print "success"
