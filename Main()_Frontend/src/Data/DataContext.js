import React, { createContext, useState, useEffect } from "react";

const DataContext = createContext();

const DataProvider = ({ children }) => {
  const [data, setData] = useState(() => {
    const savedData = localStorage.getItem("data");
    return savedData
      ? JSON.parse(savedData)
      : {
          mapcameraprovideoinfo: [], // 카메라 정보
          camnameinfo: [], // 카메라 이름 정보
          userId: "", // 사용자 ID
          userName: "", // 사용자 이름
          filterid: 0, // 필터 ID
          personInfo: [], // 새로운 person 정보 저장 (엔드포인트에서 받아오는 데이터)
        };
  });

  useEffect(() => {
    console.log("DataContext에 저장된 데이터:", data);
  }, [data]);

  useEffect(() => {
    localStorage.setItem("data", JSON.stringify(data));
  }, [data]);

  const updateData = (newData) => {
    setData((prevData) => ({
      ...prevData,
      ...newData,
      camnameinfo: Array.isArray(newData.camnameinfo)
        ? newData.camnameinfo
        : prevData.camnameinfo,
      personInfo: Array.isArray(newData.personInfo)
        ? newData.personInfo
        : prevData.personInfo,
      mapcameraprovideoinfo: Array.isArray(newData.mapcameraprovideoinfo)
        ? newData.mapcameraprovideoinfo
        : prevData.mapcameraprovideoinfo,
    }));
  };

  const clearData = () => {
    setData({
      mapcameraprovideoinfo: [],
      camnameinfo: [],
      userId: "",
      userName: "",
      filterid: 0,
      personInfo: [], // person 정보 초기화
    });
  };

  return (
    <DataContext.Provider
      value={{
        data,
        setData: updateData,
        clearData,
      }}
    >
      {children}
    </DataContext.Provider>
  );
};

export { DataContext, DataProvider };
