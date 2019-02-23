def Read():
    # Đợi người dùng nhập số lớp 
    layerCountStr = input("Vui long nhap so lop: ")

    try: 
        # Chuyển chuỗi nhập sang kiểu số nguyên  
        layerCount = int(layerCountStr) 
        # Báo lỗi khi không phải số tự nhiên khác 0
        if layerCount != float(layerCountStr) or layerCount <= 0:
            raise Exception()
    except:
        print("So khong hop le!")
        # Lặp lại việc hỏi số lớp (lặp lại hàm) 
        return Read()

    listNodeCounts = []

    i = 0
    # Lặp lại từ i -> layerCount 
    while i < layerCount:
        nodeCountStr = input("Nhap so node o layer %d: " % i)

        try:
            nodeCount = int(nodeCountStr) 
            # Báo lỗi khi không phải số tự nhiên khác 0
            if nodeCount != float(nodeCountStr) or nodeCount <= 0:
                raise Exception()
            # Thêm vào danh sách 
            listNodeCounts.append(nodeCount)
        except:
            print("So node khong hop le!")
            continue

        i = i + 1

    # Trả về một object kiểu Dict. Lấy dữ liệu như từ điển, như object["layerCount"]
    return {"layerCount": layerCount, "nodeCounts": listNodeCounts}