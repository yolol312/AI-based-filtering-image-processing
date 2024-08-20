import React, { createContext, useState, useEffect } from "react";

const RealTimeDataContext = createContext();

const RealTimeDataProvider = ({ children }) => {
  const [RealTimeData, setRealTimeData] = useState(() => {
    const savedData = localStorage.getItem("realTimeData");
    return savedData
      ? JSON.parse(savedData)
      : {
          cam_name: "", // cam_name을 추가
          cameras: [], // 카메라와 관련된 데이터를 저장하는 배열
        };
  });

  useEffect(() => {
    console.log("RealTimeDataContext에 저장된 데이터:", RealTimeData);
  }, [RealTimeData]);

  useEffect(() => {
    localStorage.setItem("realTimeData", JSON.stringify(RealTimeData));
  }, [RealTimeData]);

  const updateRealTimeData = (newData) => {
    setRealTimeData((prevData) => ({
      ...prevData,
      cam_name: newData.cam_name || prevData.cam_name, // cam_name 업데이트
      cameras: newData.cameras || prevData.cameras,
    }));
  };

  const clearRealTimeData = () => {
    setRealTimeData({
      cam_name: "", // cam_name 초기화
      cameras: [],
    });
  };

  return (
    <RealTimeDataContext.Provider
      value={{
        RealTimeData,
        setRealTimeData: updateRealTimeData,
        clearRealTimeData,
      }}
    >
      {children}
    </RealTimeDataContext.Provider>
  );
};

export { RealTimeDataContext, RealTimeDataProvider };
