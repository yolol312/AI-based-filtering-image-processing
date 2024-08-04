// DataContext.js
import React, { createContext, useState, useEffect } from 'react';

const DataContext = createContext();

const DataProvider = ({ children }) => {
  const [data, setData] = useState(() => {
    const savedData = localStorage.getItem('data');
    return savedData
      ? JSON.parse(savedData)
      : {
          mapCameraInfo: [],
          provideopersoninfo: [],
          camnameinfo: [],
          userId: '',
          userName: '',
        };
  });

  useEffect(() => {
    console.log('DataContext에 저장된 데이터:', data);
  }, [data]);

  useEffect(() => {
    localStorage.setItem('data', JSON.stringify(data));
  }, [data]);

  const updateData = (newData) => {
    setData((prevData) => ({
      ...prevData,
      ...newData,
      mapCameraInfo: Array.isArray(newData.mapCameraInfo) ? newData.mapCameraInfo : prevData.mapCameraInfo,
      provideopersoninfo: Array.isArray(newData.provideopersoninfo) ? newData.provideopersoninfo : prevData.provideopersoninfo,
      camnameinfo: Array.isArray(newData.camnameinfo) ? newData.camnameinfo : prevData.camnameinfo,
    }));
  };

  const clearData = () => {
    setData({
      mapCameraInfo: [],
      provideopersoninfo: [],
      camnameinfo: [],
      userId: '',
      userName: '',
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
