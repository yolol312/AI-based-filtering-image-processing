import React, { createContext, useState, useEffect } from "react";

const MapDataContext = createContext();

const MapDataProvider = ({ children }) => {
  const [MapInfo, setMapInfo] = useState(() => {
    const savedData = localStorage.getItem("MapInfo");
    return savedData
      ? JSON.parse(savedData)
      : {
          order_data: [],
          radius: 0,
          last_camera_name: "",
          last_camera_latitude: 0,
          last_camera_longitude: 0,
          address: "",
        };
  });
  const [triggerRouteAnimation, setTriggerRouteAnimation] = useState(false);

  useEffect(() => {
    // console.log('MapDataContext에 저장된 데이터:', MapInfo);
  }, [MapInfo]);

  useEffect(() => {
    localStorage.setItem("MapInfo", JSON.stringify(MapInfo));
  }, [MapInfo]);

  const updateMapInfo = (newMapInfo) => {
    setMapInfo(newMapInfo);
    setTriggerRouteAnimation(true); // Trigger animation when map info is updated
  };

  const clearMapInfo = () => {
    setMapInfo({
      order_data: [],
      radius: 0,
      last_camera_name: "",
      last_camera_latitude: 0,
      last_camera_longitude: 0,
      address: "",
    });
  };

  const resetRouteAnimation = () => {
    setTriggerRouteAnimation(false);
  };

  const isInitialMapInfo = () => {
    return (
      MapInfo.order_data.length === 0 &&
      MapInfo.radius === 0 &&
      MapInfo.last_camera_name === "" &&
      MapInfo.last_camera_latitude === 0 &&
      MapInfo.last_camera_longitude === 0 &&
      MapInfo.address === ""
    );
  };

  return (
    <MapDataContext.Provider
      value={{
        MapInfo,
        updateMapInfo,
        clearMapInfo,
        triggerRouteAnimation,
        resetRouteAnimation,
        isInitialMapInfo,
      }}
    >
      {children}
    </MapDataContext.Provider>
  );
};

export { MapDataContext, MapDataProvider };
