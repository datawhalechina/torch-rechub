document.querySelectorAll('.md-typeset img').forEach(item => {
    item.addEventListener('click', function () {
        // 判断图片是否具有 no-zoom 类
        if (this.classList.contains('no-zoom')) {
            return; // 如果是，跳过放大操作
        }

        // 创建遮罩
        const overlay = document.createElement('div');
        overlay.classList.add('overlay');
        document.body.appendChild(overlay);

        // 创建放大图片容器
        const zoomContainer = document.createElement('div');
        zoomContainer.classList.add('image-zoom-container');
        document.body.appendChild(zoomContainer);

        // 创建关闭按钮
        const closeButton = document.createElement('button');
        closeButton.innerText = '×'; // 设置按钮文本为 ×
        closeButton.classList.add('close-button');
        document.body.appendChild(closeButton);

        const imgSrc = this.src; // 获取点击图片的src
        const zoomedImage = document.createElement('img');
        zoomedImage.src = imgSrc;

        // 插入新的放大图片
        zoomContainer.innerHTML = '';
        zoomContainer.appendChild(zoomedImage);

        // 显示遮罩、放大图片容器和关闭按钮
        overlay.classList.add('show-overlay');
        zoomContainer.classList.add('show-overlay');
        closeButton.classList.add('show-button');

        // 初始化缩放比例和偏移量
        let scale = 1;
        let posX = 0, posY = 0;
        let isDragging = false;
        let startX, startY;

        // 阻止图片拖动的默认行为
        zoomedImage.addEventListener('dragstart', function (event) {
            event.preventDefault();
        });

        // 添加鼠标滚轮事件
        zoomedImage.addEventListener('wheel', function (event) {
            event.preventDefault();
            scale += event.deltaY * -0.01;
            scale = Math.min(Math.max(.125, scale), 4); // 限制缩放比例在0.125到4之间
            zoomedImage.style.transform = `scale(${scale}) translate(${posX}px, ${posY}px)`;
        });

        // 添加手指缩放事件
        let initialDistance = 0;
        let initialScale = scale;

        zoomedImage.addEventListener('touchstart', function (event) {
            if (event.touches.length === 2) {
                initialDistance = Math.hypot(
                    event.touches[0].pageX - event.touches[1].pageX,
                    event.touches[0].pageY - event.touches[1].pageY
                );
                initialScale = scale;
            } else if (event.touches.length === 1) {
                isDragging = true;
                startX = event.touches[0].clientX - posX;
                startY = event.touches[0].clientY - posY;
            }
        });

        zoomedImage.addEventListener('touchmove', function (event) {
            if (event.touches.length === 2) {
                event.preventDefault();
                const currentDistance = Math.hypot(
                    event.touches[0].pageX - event.touches[1].pageX,
                    event.touches[0].pageY - event.touches[1].pageY
                );
                scale = initialScale * (currentDistance / initialDistance);
                scale = Math.min(Math.max(.125, scale), 4); // 限制缩放比例在0.125到4之间
                zoomedImage.style.transform = `scale(${scale}) translate(${posX}px, ${posY}px)`;
            } else if (event.touches.length === 1 && isDragging) {
                event.preventDefault();
                posX = event.touches[0].clientX - startX;
                posY = event.touches[0].clientY - startY;
                zoomedImage.style.transform = `scale(${scale}) translate(${posX}px, ${posY}px)`;
            }
        });

        zoomedImage.addEventListener('touchend', function () {
            isDragging = false;
        });

        // 添加鼠标拖动事件
        zoomedImage.addEventListener('mousedown', function (event) {
            isDragging = true;
            startX = event.clientX - posX;
            startY = event.clientY - posY;
            zoomedImage.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', function (event) {
            if (isDragging) {
                posX = event.clientX - startX;
                posY = event.clientY - startY;
                zoomedImage.style.transform = `scale(${scale}) translate(${posX}px, ${posY}px)`;
            }
        });

        document.addEventListener('mouseup', function () {
            isDragging = false;
            zoomedImage.style.cursor = 'grab';
        });

        // 添加点击遮罩时隐藏放大图片的功能
        overlay.addEventListener('click', function () {
            document.body.removeChild(overlay);
            document.body.removeChild(zoomContainer);
            document.body.removeChild(closeButton);
        });

        // 添加点击关闭按钮时隐藏放大图片的功能
        closeButton.addEventListener('click', function () {
            document.body.removeChild(overlay);
            document.body.removeChild(zoomContainer);
            document.body.removeChild(closeButton);
        });
    });
});
