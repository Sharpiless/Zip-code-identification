import matplotlib.pyplot as plt


def final_present(post_card, numbers, two, block_number):

    # 可视化处理
    print('预测邮编为:{}\n'.format(post_card))

    figure, _ = plt.subplots()

    for i in range(block_number):
        plt.subplot(2, block_number, i+1)
        plt.imshow(numbers[i])
        plt.title(post_card[i])
        plt.xticks([]), plt.yticks([])

        plt.subplot(2, block_number, i+1+block_number)
        plt.imshow(two[i], cmap='gray')
        plt.xticks([]), plt.yticks([])

    figure.suptitle('Predicted Numbers')
    plt.show()
