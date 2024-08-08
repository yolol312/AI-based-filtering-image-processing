//DataContext.js

import React, { createContext, useState, useEffect } from 'react';

const DataContext = createContext();

const DataProvider = ({ children }) => {
  const [data, setData] = useState(() => {
    const savedData = localStorage.getItem('data');
    return savedData
      ? JSON.parse(savedData)
      : {
          mapcameraprovideoinfo: [],
          camnameinfo: [],
          userId: '',
          userName: '',
          filterid: 0,
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
      camnameinfo: Array.isArray(newData.camnameinfo) ? newData.camnameinfo : prevData.camnameinfo,
    }));
  };

  const clearData = () => {
    setData({
      mapcameraprovideoinfo: [],
      camnameinfo: [],
      userId: '',
      userName: '',
      filterid: 0,
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
