import React, { createContext, useState, useEffect } from "react";

// 컨텍스트 생성 시 기본값 설정
const ImageLogDataContext = createContext({
  imagelogInfo: [], // 초기 로그 정보는 빈 배열
  updateImageLogInfo: () => {}, // 기본적으로 아무 동작도 하지 않는 함수
  clearImageLogInfo: () => {}, // 기본적으로 아무 동작도 하지 않는 함수
});

const ImageLogDataProvider = ({ children }) => {
  const [imagelogInfo, setImageLogInfo] = useState(() => {
    try {
      const savedImageLogInfo = localStorage.getItem("imagelogInfo");
      return savedImageLogInfo ? JSON.parse(savedImageLogInfo) : [];
    } catch (error) {
      console.error("Error parsing localStorage data:", error);
      return [];
    }
  });

  useEffect(() => {
    console.log("ImageLogDataContext에 저장된 데이터:", imagelogInfo);
  }, [imagelogInfo]);

  useEffect(() => {
    localStorage.setItem("imagelogInfo", JSON.stringify(imagelogInfo));
  }, [imagelogInfo]);

  const updateImageLogInfo = (newImageLogInfo) => {
    setImageLogInfo((prevImageLogInfo) => [
      ...prevImageLogInfo,
      newImageLogInfo,
    ]);
  };

  const clearImageLogInfo = () => {
    setImageLogInfo([]);
  };

  return (
    <ImageLogDataContext.Provider
      value={{ imagelogInfo, updateImageLogInfo, clearImageLogInfo }}
    >
      {children}
    </ImageLogDataContext.Provider>
  );
};

export { ImageLogDataContext, ImageLogDataProvider };
