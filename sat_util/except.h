#pragma once
#include <string>
#include <exception>

class SatUtilException : public std::exception {
private:
    std::string _msg;
public:
    explicit SatUtilException(const std::string & msg): _msg(msg) {}
    virtual const char* what() const throw() override { return _msg.c_str(); }
};
