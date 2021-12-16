I. TỔNG QUAN VỀ HỆ THỐNG DỮ LIỆU

1. Mục đích
	- Dự đoán giá bơ trung bình của bơ "Hass" ở Mỹ
	- Xem xét mở rộng các loại trang trại Bơ đang có trong việc trồng bơ ở các vùng khác
	- Xây dựng mô hình dự báo giá trùng bình của bơ "Hass" ở Mỹ sau đó xem xét việc mở rộng sản xuất kinh doanh
2. Vi sao có dự án nào ?	
	 - Ai (Who): Doanh nghiệp là người cần
	- Tại sao (Why): Giá bơ biến động ở các vùng khác nhau ? Có nên trồng bơ các vùng đó không ?
3. Hiện tại
	- Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ có 2 loại bơ: Bơ thường và bơ hữu cơ
	- Quy cách đóng gọi theo nhiều quy chuẩn: Small/ Large/ Xlarge Bags
	- Có 3 loại item (product look up) khác nhau: 4046, 4225, 4770
4. Vấn đề	
	- Doanh nghiệp chưa có mô hình dự báo giá bơ cho việc mở rộng
	- Tối ưu sao việc tiếp cận giá bơ tới người tiêu dùng thấp nhất
5. Thách thức và cách tiếp cận - Challenge and Approach	
	- Dữ liệu được lấy trực tiếp từ máy tính tính tiền của các nhà bán lẻ dựa trên doanh số bán lẻ thực tế của bơ Hass
	- Dữ liệu đại diện cho dữ liệu lấy từ máy quét bán lẻ hàng tuần cho lượng bán lẻ (National retail volumn - units) và giá bơ từ tháng 4/2015 đến tháng 3/2018
	- Giá Trung bình (Average Price) trong bảng phản ánh giá trên một đơn vị (mỗi quả bơ), ngay cả khi nhiều đơn vị (bơ) được bán trong bao
	- Mã tra cứu sản phẩm - Product Lookup codes (PLU’s) trong bảng chỉ dành cho bơ Hass, không dành cho các sản phẩm khác.
6. Data obtained - Thu thập dữ liệu
	- Không thông quan nguồn cào data
	- Toàn bộ dữ liệu được đổ ra và lưu trữ trong tập tin avocado.csv với 18249 record.
	- Có 2 loại bơ trong tập dữ liệu và một số vùng khác nhau. Điều này cho phép chúng ta thực hiện tất cả các loại phân tích cho các vùng khác nhau hoặc phân tích toàn bộ nước mỹ theo một trong 2 loại bơ
7. Đặt ra yêu cầu với bài toán	
	
**Yêu cầu 1:** Với bài toán 1: thực hiện dự đoán giá bơ trung bình
- Thực hiện các tiền xử lý dữ liệu bổ sung (nếu cần)
- Ngoài những thuật toán regression đã được thực hiện, có thuật toán nào khác cho kết quả tốt hơn không? Thực hiện với thuật toán đó. Tổng hợp kết quả thu được."
	
**Yêu cầu 2:** Với bài toán 2: Thực hiện dự đoán giá, khả năng mở rộng trong tương lai với Organic Avocado ở vùng California
	
**Yêu cầu 3:** Hãy làm tiếp phần dự đoán giá bơ thường (Conventiton Avocado) của vùng California
	
**Yêu cầu 4:** Hãy chọn ra 1 vùng (Trong danh sách các vùng bơ "Hass" đang kinh doanh) mà bạn cho rằng trong tương lai có thể trong trọt, sản xuất kinh doanh (organic và/ hoặc Conventional Avocado). Hãy chứng minh đều này bằng cách triển khai các bài toán như đã với vùng california

II. TỔNG QUAN VỀ THỊ TRƯỜNG

1. Thị trường Hoa Kỳ
![image](https://user-images.githubusercontent.com/96172322/146219559-85c307fa-556c-4072-9b00-80432fa5d0a4.png)
2. Mục tiêu và cấn tiếp cận
![image](https://user-images.githubusercontent.com/96172322/146219587-b85199bc-f38e-4d9d-a716-9d44f183cce3.png)
3. Ai là người và cần gì ?
![image](https://user-images.githubusercontent.com/96172322/146219616-f0145c90-5c86-4d10-88cc-9381dcf29891.png)
4. Kết luận
![image](https://user-images.githubusercontent.com/96172322/146219675-8c9bb1a7-d192-411e-b309-63b4a0b3aeaa.png)

III. HƯỚNG DẪN SỬ DỤNG  VÀ CHỌN CÁC TÍNH NĂNG DỰ ĐOÁN GIÁ BƠ			
				
![image](https://user-images.githubusercontent.com/96172322/146219703-b3f3e054-c3b0-4d2e-828d-ceac84ca4a2e.png)





