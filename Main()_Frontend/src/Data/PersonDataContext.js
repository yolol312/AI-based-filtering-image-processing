import React, { createContext, useState, useEffect } from "react";

const PersonDataContext = createContext();

const PersonDataProvider = ({ children }) => {
  const [personData, setPersonData] = useState(() => {
    const savedPersonData = localStorage.getItem("personData");
    if (savedPersonData) {
      try {
        return JSON.parse(savedPersonData);
      } catch (error) {
        console.error("Failed to parse personData from localStorage:", error);
        return [];
      }
    }
    return [];
  });

  const [filterId, setFilterId] = useState(() => {
    const savedFilterId = localStorage.getItem("filterId");
    return savedFilterId || null;
  });

  useEffect(() => {
    //console.log("PersonDataContext에 저장된 데이터:", personData);
  }, [personData]);

  useEffect(() => {
    try {
      localStorage.setItem("personData", JSON.stringify(personData));
    } catch (error) {
      console.error("Failed to save personData to localStorage:", error);
    }
  }, [personData]);

  useEffect(() => {
    try {
      localStorage.setItem("filterId", filterId);
    } catch (error) {
      console.error("Failed to save filterId to localStorage:", error);
    }
  }, [filterId]);

  const updatePersonData = (newData) => {
    setPersonData(newData);
  };

  const updateFilterId = (newFilterId) => {
    setFilterId(newFilterId);
  };

  const clearPersonData = () => {
    setPersonData([]);
    setFilterId(null);
  };

  return (
    <PersonDataContext.Provider
      value={{
        personData,
        filterId,
        setPersonData: updatePersonData,
        setFilterId: updateFilterId,
        clearPersonData,
      }}
    >
      {children}
    </PersonDataContext.Provider>
  );
};

export { PersonDataContext, PersonDataProvider };
