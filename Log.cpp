#include <string>
#include <exception>
#include <cstdio>

#include "Log.hpp"


namespace cbm {

    bool Log::use_exception = false;
    bool Log::use_color     = true;

    void Log::error(const std::string& s) {
	if (use_exception) {
	    throw std::runtime_error(s);
	} else {
	    if (use_color) {
		std::printf("\033[1;%dm[Error] \033[0;%dm", FG_RED, FG_DEFAULT);
	    } else {
		std::printf("[Error] ");
	    }
	    std::printf("%s\n", s.c_str());
	}
    }

    void Log::info(const std::string& s) {
	if (use_color) {
	    if (use_color) {
		std::printf("\033[1;%dm[Info] \033[0;%dm", FG_GREEN, FG_DEFAULT);
	    } else {
		std::printf("[Info] ");
	    }
	    std::printf("%s\n", s.c_str());
	}
    }
    
    void Log::warn(const std::string& s) {
	    if (use_color) {
		std::printf("\033[1;%dm[Warn] \033[0;%dm", FG_YELLOW, FG_DEFAULT);
	    } else {
		std::printf("[Warn] ");
	    }
	    std::printf("%s\n", s.c_str());
    }

    
}  // namespace cbm
